import os
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import hashlib

def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

class TrainingDataProcessor:
    def __init__(self, **kwargs):
        """
        Data processor for your e-commerce search DCN model.

        Required files in directory (without embeddings):
            - samples.sessionreindex.jsonl      (no query_emb)
            - users_fearures.reindex.jsonl
            - corpus_.reindex.jsonl             (no title_emb)

        Pre-computed files in the same directory:
            - query_emb.npy
            - session_id2idx.json
            - item_title_emb.npy
            - item_id2idx.json
        """
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('batch_size', 128)
        self.max_history_len = kwargs.get('max_history_len', 20)
        self.valid_ratio = kwargs.get('valid_ratio', 0.01)
        self.seed = kwargs.get('seed', 42)

        # JSONL file paths
        self.samples_file = os.path.join(self.data_path,'rank.jsonl')
        self.user_feat_file = os.path.join(self.data_path, "users.jsonl")
        self.corpus_file = os.path.join(self.data_path, "corpus.jsonl")

        # Embedding & mapping file paths (in the same data_path)
        self.item_emb_path = os.path.join('./data', "item_title_emb.npy")
        self.query_emb_path = os.path.join('./data', "query_emb.npy")
        self.item_id2idx_path = os.path.join('./data', "item_id2idx.json")
        self.session_id2idx_path = os.path.join('./data', "session_id2idx.json")

        # ====== 1. Load embeddings and mappings (memmap) ======
        self._load_embeddings()

        # ====== 2. Load structured JSON data ======
        self._load_data()


    def _load_embeddings(self):
        # Load with memmap mode, not loading all into memory at once
        self.item_emb = np.load(self.item_emb_path, mmap_mode="r")   # [Ni, D]
        self.query_emb = np.load(self.query_emb_path, mmap_mode="r") # [Nq, D]

        # item_id -> idx
        with open(self.item_id2idx_path, "r") as f:
            # JSON keys are strings, convert to int for convenient item_id lookup
            self.item_id2idx = {int(k): v for k, v in json.load(f).items()}

        # session_id -> idx (session_id is usually string, keep as str)
        with open(self.session_id2idx_path, "r") as f:
            self.session_id2idx = json.load(f)

        self.item_emb_dim = self.item_emb.shape[1]
        self.query_emb_dim = self.query_emb.shape[1]

        print(f"[INFO] item_emb:  shape={self.item_emb.shape}")
        print(f"[INFO] query_emb: shape={self.query_emb.shape}")

    def _load_data(self):
        # 1. User profiles
        self.user_features = {}
        with open(self.user_feat_file, "r") as f:
            for line in tqdm(f, desc="Users"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                uid = obj["user_id"]
                self.user_features[uid] = obj

        # 2. Item information (sparse features only, no title_emb)
        self.item_corpus = {}
        with open(self.corpus_file, "r") as f:
            for line in tqdm(f, desc="Items"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                iid = obj["item_id"]
                self.item_corpus[iid] = obj


        raw_all_train = []
        raw_test_samples = []

        
        # Provide total for tqdm progress bar (optional)
        total = sum(1 for _ in open(self.samples_file, "r"))
        with open(self.samples_file, "r") as f:
            for line in tqdm(f, total=total, desc="Samples"):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
        
                split_type = obj.get("split", "train")  # Default to train
        
                # Split train and test based on 'split' field
                if split_type == "test":
                    raw_test_samples.append(obj)
                else:
                    # train or others are included in training candidate set
                    raw_all_train.append(obj)
                    
        print(f"[INFO] Loaded users   = {len(self.user_features)}")
        print(f"[INFO] Loaded items   = {len(self.item_corpus)}")
        print(f"[INFO] All-train candidates (split='train') = {len(raw_all_train)}")
        print(f"[INFO] Test samples   (split='test') = {len(raw_test_samples)}")
        
        # ===== 从 raw_all_train 中随机划分 10% 作为 valid =====
        random.seed(self.seed)
        random.shuffle(raw_all_train)
        
        valid_size = int(len(raw_all_train) * 0.10)
        raw_valid_samples = raw_all_train[:valid_size]
        raw_train_samples = raw_all_train[valid_size:]
        
        self.train_samples = raw_train_samples
        self.valid_samples = raw_valid_samples
        self.test_samples  = raw_test_samples
        
        print(f"[INFO] Final Train samples = {len(self.train_samples)}")
        print(f"[INFO] Final Valid samples = {len(self.valid_samples)} (10%)")
        print(f"[INFO] Final Test  samples = {len(self.test_samples)}")
    

    def _safe_get(self, obj, key, default=0.0):
        v = obj.get(key, default) if isinstance(obj, dict) else default
        if v is None:
            return default
        try:
            return float(v)
        except Exception:
            return default


    def get_item_sparse_features(self, item_id):
        item = self.item_corpus.get(item_id)
        if item is None:
            return {
                'category_level1_id': torch.tensor(0, dtype=torch.long),
                'category_level2_id': torch.tensor(0, dtype=torch.long),
                'category_level3_id': torch.tensor(0, dtype=torch.long),
                'item_id': torch.tensor(0, dtype=torch.long),
            }

        return {
            'category_level1_id': torch.tensor(item.get('category_level1_id', 0), dtype=torch.long),
            'category_level2_id': torch.tensor(item.get('category_level2_id', 0), dtype=torch.long),
            'category_level3_id': torch.tensor(item.get('category_level3_id', 0), dtype=torch.long),
            'item_id': torch.tensor(item_id, dtype=torch.long),
        }

    def get_item_title_emb(self, item_id):
        idx = self.item_id2idx.get(int(item_id), None)
        if idx is None:
            return None
        vec = self.item_emb[idx]                 # Read-only memmap view
        return torch.tensor(vec, dtype=torch.float32)                # Explicitly copy

    def get_user_sparse_features(self, user_id):
        user = self.user_features.get(user_id, None)

        gender_map = {'M': 0, 'F': 1}
        age_map = {
            '0-11': 0,
            '12-17': 1,
            '18-23': 2,
            '24-30': 3,
            '31-40': 4,
            '41-49': 5,
            '50+': 6,
        }

        if user is None:
            gender_idx = 2 
            age_idx = 9
        else:
            gender_idx = gender_map.get(user.get('gender'))
            age_idx = age_map.get(user.get('age'))


        return {
            'gender': torch.tensor(gender_idx, dtype=torch.long),
            'age': torch.tensor(age_idx, dtype=torch.long),
            'user_id': torch.tensor(user_id, dtype=torch.long),
        }

    def get_user_history1(self, sample):
        click_list = sample.get('recently_clicked_item_ids', []) or []
        hist = click_list[-self.max_history_len:]
        if len(hist) < self.max_history_len:
            hist = [0] * (self.max_history_len - len(hist)) + hist
        return torch.tensor(hist, dtype=torch.long)
        
    def get_user_history(self, sample):
        """
        """
        click_list = sample.get('recently_clicked_item_ids', []) or []
        click_list = click_list[-self.max_history_len:]

        hist_item = []
        hist_cat1 = []
        hist_cat2 = []
        hist_cat3 = []
        for iid in click_list:
            iid = int(iid) if iid is not None else 0
            item = self.item_corpus.get(iid)

            if item is None:
                hist_item.append(0)
                hist_cat1.append(0)
                hist_cat2.append(0)
                hist_cat3.append(0)
            else:
                hist_item.append(iid)
                hist_cat1.append(int(item.get('category_level1_id', 0) or 0))
                hist_cat2.append(int(item.get('category_level2_id', 0) or 0))
                hist_cat3.append(int(item.get('category_level3_id', 0) or 0))

        # left padding to max_history_len
        pad_len = self.max_history_len - len(hist_item)
        if pad_len > 0:
            hist_item = [0] * pad_len + hist_item
            hist_cat1 = [0] * pad_len + hist_cat1
            hist_cat2 = [0] * pad_len + hist_cat2
            hist_cat3 = [0] * pad_len + hist_cat3

        return (
            torch.tensor(hist_item, dtype=torch.long),
            torch.tensor(hist_cat1, dtype=torch.long),
            torch.tensor(hist_cat2, dtype=torch.long),
            torch.tensor(hist_cat3, dtype=torch.long),
        )


    def collate_fn(self, batch):
        # ===== 1. Query embedding: session_id -> idx -> query_emb =====
        session_ids = []
        for sample in batch:
            sid = sample.get("session_id")
            if sid is None:
                raise ValueError("sample missing session_id")
            session_ids.append(str(sid))

        query_idx = []
        for sid in session_ids:
            idx = self.session_id2idx.get(sid, None)
            if idx is None:
                raise ValueError(f"session_id={sid} not found in session_id2idx")
            query_idx.append(idx)

        query_idx = np.array(query_idx, dtype=np.int64)
        query_emb_np = self.query_emb[query_idx]           # [B, D]
        query_emb = torch.tensor(query_emb_np, dtype=torch.float32)
        query_features = {
            'query_emb': query_emb
        }

        # ===== 2. User & item & label =====
        user_gender = []
        user_age = []
        user_id_list = []
        user_history = []
        user_history_cat1 = []
        user_history_cat2 = []
        user_history_cat3 = []
        
        item_cat1 = []
        item_cat2 = []
        item_cat3 = []
        item_id_list = []
        item_title_embs = []


        labels = []

        for sample in batch:
            uid = sample['user_id']
            iid = sample['target_item_id']
            # Positive sample logic: is_clicked=1 or is_purchased=1
            is_clicked = sample.get('is_clicked', 0)
            is_purchased = sample.get('is_purchased', 0)
            lbl = 1 if (is_clicked == 1 or is_purchased == 1) else 0

            # User sparse features
            u_sparse = self.get_user_sparse_features(uid)
            user_gender.append(u_sparse['gender'])
            user_age.append(u_sparse['age'])
            user_id_list.append(u_sparse['user_id'])

            
            hist_item, hist_cat1, hist_cat2, hist_cat3 = self.get_user_history(sample)
            user_history.append(hist_item)
            user_history_cat1.append(hist_cat1)
            user_history_cat2.append(hist_cat2)
            user_history_cat3.append(hist_cat3)
            # Item sparse features
            i_sparse = self.get_item_sparse_features(iid)
            item_cat1.append(i_sparse['category_level1_id'])
            item_cat2.append(i_sparse['category_level2_id'])
            item_cat3.append(i_sparse['category_level3_id'])
            #item_seller.append(i_sparse['seller_id'])
            item_id_list.append(i_sparse['item_id'])

            # Item vector from npy
            item_emb = self.get_item_title_emb(iid)
            if item_emb is None:
                raise ValueError(f"item_id={iid} missing embedding, please confirm item_title_emb.npy / item_id2idx.json are generated")
            item_title_embs.append(item_emb)

            labels.append(float(lbl))

        user_features = {
            'gender': torch.stack(user_gender),
            'age': torch.stack(user_age),
            'user_id': torch.stack(user_id_list),

            'recent_clicked_items': torch.stack(user_history),          # [B, L]
            'recent_clicked_cat1': torch.stack(user_history_cat1),      # [B, L]
            'recent_clicked_cat2': torch.stack(user_history_cat2),      # [B, L]
            'recent_clicked_cat3': torch.stack(user_history_cat3),      # [B, L]
        }

        item_features = {
            'category_level1_id': torch.stack(item_cat1),
            'category_level2_id': torch.stack(item_cat2),
            'category_level3_id': torch.stack(item_cat3),
            'item_id': torch.stack(item_id_list),
            'title_emb': torch.stack(item_title_embs),
        }

        labels = torch.tensor(labels, dtype=torch.float32)

        return query_features, user_features, item_features, labels

    # ======================
    #     DataLoader Interface
    # ======================
    def get_train_dataloader(self):
        return DataLoader(
            self.train_samples,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

    def get_valid_dataloader(self):
        if len(self.valid_samples) == 0:
            return None
        return DataLoader(
            self.valid_samples,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

    def get_test_dataloader(self):
        if len(self.test_samples) == 0:
            return None
        return DataLoader(
            self.test_samples,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True,
        )

    # Compatible with original interface, return train dataloader if not specified
    def get_dataloader(self):
        return self.get_train_dataloader()
