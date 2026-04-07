from torch.utils.data import DataLoader
import os
import numpy as np
from typing import List, Tuple
import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class Dice(nn.Module):
    """
    DIN: Dice activation
    y = p(x) * x + (1 - p(x)) * alpha * x
    p(x) = sigmoid(BN(x))
    """
    def __init__(self, dim, eps=1e-9):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim, eps=eps)
        self.alpha = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: [B, dim]
        p = torch.sigmoid(self.bn(x))
        return p * x + (1 - p) * self.alpha * x

        
class LocalActivationUnit(nn.Module):
    """
    DIN: a(e_j, v_A) -> w_j
    Typical input: [e_j, v_A, e_j - v_A, e_j * v_A]
    Output: weight w_j (no softmax normalization by default, per DIN paper)
    """
    def __init__(self, dim: int, hidden_units=(128, 64), dropout=0.0, use_dice=False):
        super().__init__()
        layers = []
        in_dim = dim * 4
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            if use_dice:
                # Dice is optional enhancement in the paper, using PReLU/ReLU here
                layers.append(Dice(h))
            else:
                layers.append(nn.PReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))  # output weight (logit)
        self.mlp = nn.Sequential(*layers)

    def forward(self, hist_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        hist_emb:   [B, L, D]
        target_emb: [B, D]
        return:     weights [B, L, 1]  (raw weights; you decide sigmoid/softplus/etc.)
        """
        B, L, D = hist_emb.shape
        t = target_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        x = torch.cat([hist_emb, t, hist_emb - t, hist_emb * t], dim=-1)  # [B, L, 4D]
        x = x.view(B * L, -1)  # BN1d expects [N, C]
        w = self.mlp(x)        # [B*L, 1]
        w = w.view(B, L, 1)    # [B, L, 1]
        return w
        
class DINModel(nn.Module):
    def __init__(
        self,
        config,
        num_cross_layers=3,
        hidden_size=256,
        dropout_rate=0.2,
        user_id_embedding_dim=32,  
        version='v1',
    ):
        super().__init__()

        self.text_hidden_size = 512
        self.version=version
        # Sparse feature embedding layers
        self.user_embedding_dims = {
            'gender': (2, 8),          # Gender: 3 classes
            'age': (7, 16),           # Age: 10 classes
            'user_id': (102086+1, user_id_embedding_dim),  # User ID: 0-15482
        }

        self.item_embedding_dims = {
            'item_id': (6634118+1, user_id_embedding_dim),   # Note ID: 0-1983939
            'category_level1_id': (95, 16),   # Level 1 category
            'category_level2_id': (1074, 32),  # Level 2 category
            'category_level3_id': (6470, 64),  # Level 3 category
        }
        
        # Create embedding layers
        self.user_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.user_embedding_dims.items()
        })
        
        self.item_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.item_embedding_dims.items()
        })
                
        # Calculate feature dimensions
        self.query_dim = self.text_hidden_size

        self.user_sparse_dim = sum(dim[1] for dim in self.user_embedding_dims.values())
        self.item_sparse_dim = sum(dim[1] for dim in self.item_embedding_dims.values())
        
        # Total feature dimension
        self.history_dim = (
            self.item_embedding_dims['item_id'][1] +
            self.item_embedding_dims['category_level1_id'][1] +
            self.item_embedding_dims['category_level2_id'][1] +
            self.item_embedding_dims['category_level3_id'][1]
        )

        self.total_feature_dim = (
            self.query_dim +
            self.user_sparse_dim +
            self.history_dim +
            self.item_sparse_dim +
            self.text_hidden_size
        )
        
        self.din_activation = LocalActivationUnit(
            dim=self.history_dim,
            hidden_units=(128, 64),
            dropout=0.0,
            use_dice=False,   # Turn off for now, can add Dice later to reproduce paper
        )        
        # DNN layer
        self.cross_network = CrossNetwork(self.total_feature_dim, num_cross_layers,self.version)
        
        # DNN layer
        self.dnn = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size + self.total_feature_dim, 1),
        )
        
    def forward(self, query_features, user_features, item_features):
        # 1. Process query features
        query_vector = query_features['query_emb']
        
        # 2. Process user features
        user_sparse_embeddings = []
        for key, embedding_layer in self.user_embeddings.items():
            user_sparse_embeddings.append(embedding_layer(user_features[key]))
        user_sparse = torch.cat(user_sparse_embeddings, dim=-1)
        
        # Process historical behavior - mean pooling
        # ===== Historical click sequence embedding (item_id + cat1/2/3) =====
        hist_item = user_features['recent_clicked_items']     # [B, L]
        hist_c1   = user_features['recent_clicked_cat1']      # [B, L]
        hist_c2   = user_features['recent_clicked_cat2']      # [B, L]
        hist_c3   = user_features['recent_clicked_cat3']      # [B, L]
        emb_item = self.item_embeddings['item_id'](hist_item)                 # [B, L, d_item]
        emb_c1   = self.item_embeddings['category_level1_id'](hist_c1)        # [B, L, d1]
        emb_c2   = self.item_embeddings['category_level2_id'](hist_c2)        # [B, L, d2]
        emb_c3   = self.item_embeddings['category_level3_id'](hist_c3)        # [B, L, d3]
        hist_emb = torch.cat([emb_item, emb_c1, emb_c2, emb_c3], dim=-1)      # [B, L, d_sum]
        

        
        # 3. Process note features
        item_text = item_features['title_emb']
        item_sparse_embeddings = []
        for key, embedding_layer in self.item_embeddings.items():
            item_sparse_embeddings.append(embedding_layer(item_features[key]))
            
        item_sparse = torch.cat(item_sparse_embeddings, dim=-1)
        mask = (hist_item != 0).float().unsqueeze(-1)
        

        w = self.din_activation(hist_emb, item_sparse)  # [B, L, 1] directly as w_j
        w = w * mask                                    # mask padding
        history_encoding = (hist_emb * w).sum(dim=1)    # [B, D] Key: do not divide by denom

        # 4. Concatenate features
        combined_features = torch.cat([
            query_vector,
            user_sparse,
            history_encoding,
            item_sparse,
            item_text
        ], dim=-1)
        
        cross_output = self.cross_network(combined_features)
        dnn_output = self.dnn(combined_features)
        
        # 6. Merge DCN and DNN outputs
        final_output = torch.cat([cross_output, dnn_output], dim=-1)
        
        # 7. Output layer
        logits = self.output_layer(final_output)
        return logits.squeeze(-1)
    
    def get_loss(self, query_features, user_features, item_features, labels):
        predictions = self(query_features, user_features, item_features)
        return torch.nn.BCEWithLogitsLoss()(predictions, labels)
        
    def load_model(self, model_path):
        """Load model parameters from specified path
        
        Args:
            model_path (str): Path to model parameter file
        """            
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model parameters from {model_path}")
        except Exception as e:
            print(f"Error loading model parameters: {str(e)}")
            print('Reinitializing model parameters')


class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers, version="v1"):
        super().__init__()
        self.num_layers = num_layers
        self.version = version
        assert version in ("v1", "v2")

        if self.version == "v1":
            # Original DCN: rank-1 cross
            self.weights = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim))
                for _ in range(num_layers)
            ])
            self.bias = nn.ParameterList([
                nn.Parameter(torch.zeros(input_dim))
                for _ in range(num_layers)
            ])
        else:
            # DCN-V2: full-rank cross, W ∈ R^{d×d}
            self.W_list = nn.ParameterList([
                nn.Parameter(torch.randn(input_dim, input_dim))
                for _ in range(num_layers)
            ])
            self.bias = nn.ParameterList([
                nn.Parameter(torch.zeros(input_dim))
                for _ in range(num_layers)
            ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(input_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        x: [B, d]
        """
        x0 = x
        xi = x

        for i in range(self.num_layers):
            if self.version == "v1":
                # DCN-V1:
                # x_{l+1} = x0 * (w^T x_l) + b + x_l
                xw = torch.sum(xi * self.weights[i], dim=-1, keepdim=True)  # [B, 1]
                cross = x0 * xw + self.bias[i]                              # [B, d]
                xi = xi + cross
            else:
                # DCN-V2:
                # x_{l+1} = x0 ⊙ (W x_l + b) + x_l
                cross = torch.matmul(xi, self.W_list[i]) + self.bias[i]     # [B, d]
                cross = x0 * cross                                          # [B, d]
                xi = xi + cross

            # BN normalizes the output of each layer [B, d]
            xi = self.batch_norms[i](xi)

        return xi



        
class DCNModel(nn.Module):
    def __init__(
        self,
        config,
        num_cross_layers=3,
        hidden_size=256,
        dropout_rate=0.2,
        user_id_embedding_dim=32,  
        version='v1',
    ):
        super().__init__()

        self.text_hidden_size = 512
        self.version=version
        # Sparse feature embedding layers
        self.user_embedding_dims = {
            'gender': (2, 8),          # Gender: 3 classes
            'age': (7, 16),           # Age: 10 classes
            'user_id': (102086+1, user_id_embedding_dim),  # User ID: 0-15482
        }

        self.item_embedding_dims = {
            'item_id': (6634118+1, user_id_embedding_dim),   # Note ID: 0-1983939
            'category_level1_id': (95, 16),   # Level 1 category
            'category_level2_id': (1074, 32),  # Level 2 category
            'category_level3_id': (6470, 64),  # Level 3 category

        }
        
        # Create embedding layers
        self.user_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.user_embedding_dims.items()
        })
        
        self.item_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.item_embedding_dims.items()
        })
                
        # Calculate feature dimensions
        self.query_dim = self.text_hidden_size

        self.user_sparse_dim = sum(dim[1] for dim in self.user_embedding_dims.values())
        self.item_sparse_dim = sum(dim[1] for dim in self.item_embedding_dims.values())
        
        # Total feature dimension
        self.history_dim = (
            self.item_embedding_dims['item_id'][1] +
            self.item_embedding_dims['category_level1_id'][1] +
            self.item_embedding_dims['category_level2_id'][1] +
            self.item_embedding_dims['category_level3_id'][1] #+
        )

        self.total_feature_dim = (
            self.query_dim +
            self.user_sparse_dim +
            self.history_dim +
            self.item_sparse_dim +
            self.text_hidden_size
        )
        
        # DCN layer
        self.cross_network = CrossNetwork(self.total_feature_dim, num_cross_layers,self.version)
        
        # DNN layer
        self.dnn = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size + self.total_feature_dim, 1),
        )
        
    def forward(self, query_features, user_features, item_features):
        # 1. Process query features
        query_vector = query_features['query_emb']
        
        # 2. Process user features
        user_sparse_embeddings = []
        for key, embedding_layer in self.user_embeddings.items():
            user_sparse_embeddings.append(embedding_layer(user_features[key]))
        user_sparse = torch.cat(user_sparse_embeddings, dim=-1)
        
        # Process historical behavior - mean pooling
        hist_item = user_features['recent_clicked_items']     # [B, L]
        hist_c1   = user_features['recent_clicked_cat1']      # [B, L]
        hist_c2   = user_features['recent_clicked_cat2']      # [B, L]
        hist_c3   = user_features['recent_clicked_cat3']      # [B, L]
        emb_item = self.item_embeddings['item_id'](hist_item)                 # [B, L, d_item]
        emb_c1   = self.item_embeddings['category_level1_id'](hist_c1)        # [B, L, d1]
        emb_c2   = self.item_embeddings['category_level2_id'](hist_c2)        # [B, L, d2]
        emb_c3   = self.item_embeddings['category_level3_id'](hist_c3)        # [B, L, d3]
        hist_emb = torch.cat([emb_item, emb_c1, emb_c2, emb_c3], dim=-1)      # [B, L, d_sum]

        # masked mean pooling（把 padding=0 的位置排除）
        mask = (hist_item != 0).float().unsqueeze(-1)                         # [B, L, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)                                # [B, 1]
        history_encoding = (hist_emb * mask).sum(dim=1) / denom               # [B, d_sum]

        
        # 3. Process note features
        item_text = item_features['title_emb']
        item_sparse_embeddings = []
        for key, embedding_layer in self.item_embeddings.items():
            item_sparse_embeddings.append(embedding_layer(item_features[key]))
        item_sparse = torch.cat(item_sparse_embeddings, dim=-1)
        
        # 4. Concatenate features
        combined_features = torch.cat([
            query_vector,
            user_sparse,
            history_encoding,
            item_sparse,
            item_text
        ], dim=-1)
    
        # 5. DCN processing
        cross_output = self.cross_network(combined_features)
        dnn_output = self.dnn(combined_features)
        
        # 6. Merge DCN and DNN outputs
        final_output = torch.cat([cross_output, dnn_output], dim=-1)
        
        # 7. Output layer
        logits = self.output_layer(final_output)
        return logits.squeeze(-1)
    
    def get_loss(self, query_features, user_features, item_features, labels):
        predictions = self(query_features, user_features, item_features)
        return torch.nn.BCEWithLogitsLoss()(predictions, labels)
        
    def load_model(self, model_path):
        """Load model parameters from specified path
        
        Args:
            model_path (str): Path to model parameter file
        """            
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model parameters from {model_path}")
        except Exception as e:
            print(f"Error loading model parameters: {str(e)}")
            print('Reinitializing model parameters')




        
class DNNModel(nn.Module):
    def __init__(
        self,
        config,
        num_cross_layers=3,
        hidden_size=256,
        dropout_rate=0.2,
        user_id_embedding_dim=32,  
        version='v1',
    ):
        super().__init__()

        self.text_hidden_size = 512
        self.version=version
        # Sparse feature embedding layers
        self.user_embedding_dims = {
            'gender': (2, 8),          # Gender: 3 classes
            'age': (7, 16),           # Age: 10 classes
            'user_id': (102086+1, user_id_embedding_dim),  # User ID: 0-15482
        }

        self.item_embedding_dims = {
            'item_id': (6634118+1, user_id_embedding_dim),   # Note ID: 0-1983939
            'category_level1_id': (95, 16),   # Level 1 category
            'category_level2_id': (1074, 32),  # Level 2 category
            'category_level3_id': (6470, 64),  # Level 3 category

        }
        
        # Create embedding layers
        self.user_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.user_embedding_dims.items()
        })
        
        self.item_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.item_embedding_dims.items()
        })
                
        # Calculate feature dimensions
        self.query_dim = self.text_hidden_size

        self.user_sparse_dim = sum(dim[1] for dim in self.user_embedding_dims.values())
        self.item_sparse_dim = sum(dim[1] for dim in self.item_embedding_dims.values())
        
        # Total feature dimension
        self.history_dim = (
            self.item_embedding_dims['item_id'][1] +
            self.item_embedding_dims['category_level1_id'][1] +
            self.item_embedding_dims['category_level2_id'][1] +
            self.item_embedding_dims['category_level3_id'][1] 
        )

        self.total_feature_dim = (
            self.query_dim +
            self.user_sparse_dim +
            self.history_dim +
            self.item_sparse_dim +
            self.text_hidden_size
        )
        
        # DNN layer
        self.dnn = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 1),
        )
        
    def forward(self, query_features, user_features, item_features):
        # 1. Process query features
        query_vector = query_features['query_emb']
        
        # 2. Process user features
        user_sparse_embeddings = []
        for key, embedding_layer in self.user_embeddings.items():
            user_sparse_embeddings.append(embedding_layer(user_features[key]))
        user_sparse = torch.cat(user_sparse_embeddings, dim=-1)
        
        # Process historical behavior - mean pooling
        hist_item = user_features['recent_clicked_items']     # [B, L]
        hist_c1   = user_features['recent_clicked_cat1']      # [B, L]
        hist_c2   = user_features['recent_clicked_cat2']      # [B, L]
        hist_c3   = user_features['recent_clicked_cat3']      # [B, L]
        emb_item = self.item_embeddings['item_id'](hist_item)                 # [B, L, d_item]
        emb_c1   = self.item_embeddings['category_level1_id'](hist_c1)        # [B, L, d1]
        emb_c2   = self.item_embeddings['category_level2_id'](hist_c2)        # [B, L, d2]
        emb_c3   = self.item_embeddings['category_level3_id'](hist_c3)        # [B, L, d3]
        hist_emb = torch.cat([emb_item, emb_c1, emb_c2, emb_c3], dim=-1)      # [B, L, d_sum]

        # masked mean pooling（把 padding=0 的位置排除）
        mask = (hist_item != 0).float().unsqueeze(-1)                         # [B, L, 1]
        denom = mask.sum(dim=1).clamp_min(1.0)                                # [B, 1]
        history_encoding = (hist_emb * mask).sum(dim=1) / denom               # [B, d_sum]

        
        # 3. Process note features
        item_text = item_features['title_emb']
        item_sparse_embeddings = []
        for key, embedding_layer in self.item_embeddings.items():
            item_sparse_embeddings.append(embedding_layer(item_features[key]))
        item_sparse = torch.cat(item_sparse_embeddings, dim=-1)
        
        # 4. Concatenate features
        combined_features = torch.cat([
            query_vector,
            user_sparse,
            history_encoding,
            item_sparse,
            item_text
        ], dim=-1)
        
        # 5. DCN processing
        dnn_output = self.dnn(combined_features)
        
        # 6. Merge DCN and DNN outputs
        
        # 7. Output layer
        logits = self.output_layer(dnn_output)
        return logits.squeeze(-1)
    
    def get_loss(self, query_features, user_features, item_features, labels):
        predictions = self(query_features, user_features, item_features)
        return torch.nn.BCEWithLogitsLoss()(predictions, labels)
        
    def load_model(self, model_path):
        """Load model parameters from specified path
        
        Args:
            model_path (str): Path to model parameter file
        """            
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model parameters from {model_path}")
        except Exception as e:
            print(f"Error loading model parameters: {str(e)}")
            print('Reinitializing model parameters')



class WideDeepModel(nn.Module):
    def __init__(
        self,
        config,
        num_cross_layers=3,
        hidden_size=256,
        dropout_rate=0.2,
        user_id_embedding_dim=32,   
        version='v1',
    ):
        super().__init__()

        self.text_hidden_size = 512
        self.version = version
        
        # ==========================================
        # 1. Feature Dimension Definitions
        # ==========================================
        self.user_embedding_dims = {
            'gender': (2, 8),           
            'age': (7, 16),            
            'user_id': (102086+1, user_id_embedding_dim), 
        }

        self.item_embedding_dims = {
            'item_id': (6634118+1, user_id_embedding_dim),   
            'category_level1_id': (95, 16),    
            'category_level2_id': (1074, 32),  
            'category_level3_id': (6470, 64),  
        }
        
        # ==========================================
        # 2. Wide Component (The Wide Component)
        # Mechanism: Linear model y = w*x + b 
        # Implementation: Use Embedding with output_dim=1 to simulate linear weights w
        # Note: In practice, Wide component usually contains manually crafted cross features (Cross Product)
        # Here for demonstration, we directly apply linear "memorization" to sparse ID features
        # ==========================================
        self.wide_user_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], 1) for k, dim in self.user_embedding_dims.items()
        })
        self.wide_item_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], 1) for k, dim in self.item_embedding_dims.items()
        })
        # Wide component's Bias
        self.wide_bias = nn.Parameter(torch.zeros(1))

        # ==========================================
        # 3. Deep Component (The Deep Component)
        # Mechanism: Feed-forward NN with Embeddings 
        # ==========================================
        
        self.deep_user_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.user_embedding_dims.items()
        })
        
        self.deep_item_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.item_embedding_dims.items()
        })
                
        # Deep - Calculate dimensions
        self.query_dim = self.text_hidden_size
        self.user_sparse_dim = sum(dim[1] for dim in self.user_embedding_dims.values())
        self.item_sparse_dim = sum(dim[1] for dim in self.item_embedding_dims.values())
        
        self.history_dim = (
            self.item_embedding_dims['item_id'][1] +
            self.item_embedding_dims['category_level1_id'][1] +
            self.item_embedding_dims['category_level2_id'][1] +
            self.item_embedding_dims['category_level3_id'][1]
        )

        self.total_feature_dim = (
            self.query_dim +
            self.user_sparse_dim +
            self.history_dim +
            self.item_sparse_dim +
            self.text_hidden_size
        )
        
        self.dnn = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Deep - Output Unit (Logits)
        self.deep_output_layer = nn.Linear(hidden_size, 1)
        
    def forward(self, query_features, user_features, item_features):
        # ==========================================
        # A. Wide Component Forward
        # Logic: Sum(Linear Weights of features) + Bias [cite: 128]
        # ==========================================
        wide_logits = 0
        
        # Accumulate User feature linear weights
        for key, embedding_layer in self.wide_user_embeddings.items():
            # embedding output is [B, 1], squeeze to [B]
            wide_logits += embedding_layer(user_features[key]).squeeze(-1)
            
        # Accumulate Item feature linear weights
        for key, embedding_layer in self.wide_item_embeddings.items():
            wide_logits += embedding_layer(item_features[key]).squeeze(-1)
            
        # Add Bias
        wide_logits += self.wide_bias

        # 1. Process query
        query_vector = query_features['query_emb']
        
        # 2. Process user sparse
        user_sparse_list = [
            self.deep_user_embeddings[k](user_features[k]) 
            for k in self.user_embedding_dims
        ]
        user_sparse = torch.cat(user_sparse_list, dim=-1)
        
        # 3. Process history (Mean Pooling)
        hist_item = user_features['recent_clicked_items']
        hist_c1   = user_features['recent_clicked_cat1']
        hist_c2   = user_features['recent_clicked_cat2']
        hist_c3   = user_features['recent_clicked_cat3']

        emb_item = self.deep_item_embeddings['item_id'](hist_item)
        emb_c1   = self.deep_item_embeddings['category_level1_id'](hist_c1)
        emb_c2   = self.deep_item_embeddings['category_level2_id'](hist_c2)
        emb_c3   = self.deep_item_embeddings['category_level3_id'](hist_c3)
        
        hist_emb = torch.cat([emb_item, emb_c1, emb_c2, emb_c3], dim=-1)
        
        mask = (hist_item != 0).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        history_encoding = (hist_emb * mask).sum(dim=1) / denom

        # 4. Process item sparse & text
        item_text = item_features['title_emb']
        item_sparse_list = [
            self.deep_item_embeddings[k](item_features[k]) 
            for k in self.item_embedding_dims
        ]
        item_sparse = torch.cat(item_sparse_list, dim=-1)
        
        # 5. Concatenate & MLP
        combined_features = torch.cat([
            query_vector,
            user_sparse,
            history_encoding,
            item_sparse,
            item_text
        ], dim=-1)
        
        dnn_output = self.dnn(combined_features)
        deep_logits = self.deep_output_layer(dnn_output).squeeze(-1)
        
        # ==========================================
        # C. Joint Training Output [cite: 128]
        # Logic: P(Y=1|x) = Sigmoid(Wide_Logits + Deep_Logits)
        # ==========================================
        final_logits = wide_logits + deep_logits
        
        return final_logits

    def get_loss(self, query_features, user_features, item_features, labels):
        predictions = self(query_features, user_features, item_features)
        return torch.nn.BCEWithLogitsLoss()(predictions, labels)
        
    def load_model(self, model_path):
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model parameters from {model_path}")
        except Exception as e:
            print(f"Error loading model parameters: {str(e)}")
            print('Reinitializing model parameters')
