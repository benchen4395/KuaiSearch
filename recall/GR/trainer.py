from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
from torch import nn
import torch.nn.functional as F
import time
import torch
import gc

class IndexingTrainer(Trainer):
    def __init__(self, restrict_decode_vocab, logits_processor=None, mode='gen', **kwds):
        super().__init__(**kwds)
        self.restrict_decode_vocab = restrict_decode_vocab 
        self.logits_processor = logits_processor
        self.mode = mode
        self.margin = 0.25 
        self.Lambda = 1000

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.mode == 'gen':
            loss = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=inputs['labels']).loss
            if return_outputs:
                return loss, [None, None]  # fake outputs
            return loss
        elif self.mode == 'rank':
            B, L = inputs['labels'].shape
            input_ids = inputs['input_ids']      # [B, L]
            attention_mask = inputs['attention_mask']  # [B, L]
            labels = inputs['labels']            # [B, L]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            gen_loss = outputs.loss
            logits = outputs.logits  # [B, L, V]
            log_probs = F.log_softmax(logits, dim=-1)  # [B, L, vocab]
            safe_labels = labels.clone()
            safe_labels[safe_labels == -100] = 0  
            label_log_probs = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, L]
            mask = (labels != -100).float()
            pos_scores = (label_log_probs * mask).sum(dim=1)

            valid_index = self.restrict_decode_vocab(None, None)  # list of allowed token ids
            log_probs_valid = log_probs[:, :, valid_index]  # [B, L, N_valid]
            L_valid = log_probs_valid.shape[2]

            rand_token_idx = torch.randint(0, L_valid, (B, L), device=logits.device)  # [B, L]
            rand_token_log_probs = log_probs_valid.gather(2, rand_token_idx.unsqueeze(-1)).squeeze(-1)  # [B, L]
            rand_neg_scores = (rand_token_log_probs * mask).sum(dim=1)  # [B]
            hardest_neg_scores = torch.max(log_probs_valid, dim=2)[0].sum(1)  # [B]

            lrank1 = F.relu(hardest_neg_scores - pos_scores + self.margin).mean()
            lrank2 = F.relu(rand_neg_scores - pos_scores + self.margin).mean()
            loss = lrank1 + lrank2 + self.Lambda * gen_loss

            if return_outputs:
                return loss, [None, None]
            return loss

    def log(self, *logs):
        super().log(*logs)
        max_history = 100
        if len(self.state.log_history) > max_history:
            self.state.log_history = self.state.log_history[-max_history:]


    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        model.eval()

        with torch.no_grad():
            # greedy search
            if prediction_loss_only:
                loss = self.compute_loss(model, inputs)
                return (loss.detach(), None, None)
            doc_ids = model.generate(
                inputs['input_ids'].to(self.args.device),
                max_new_tokens = 20,
                logits_processor=self.logits_processor,
                num_beams=1,
                do_sample=False,
                )
        return (None, doc_ids, inputs['labels'])

