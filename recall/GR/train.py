from transformers import LogitsProcessor
from dataset import IndexingTrainDataset, IndexingCollator, QueryEvalCollator
from transformers import MT5Tokenizer, T5Tokenizer, MT5ForConditionalGeneration, TrainingArguments, TrainerCallback, MT5Config
from transformers import AutoTokenizer, AutoModelForCausalLM, EarlyStoppingCallback
from trainer import IndexingTrainer
import numpy as np
import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
import logging
import time
import gc
import os
import json
import pickle
import wandb 
logging.getLogger("transformers").setLevel(logging.INFO)
logging.getLogger("datasets").setLevel(logging.INFO)
class RestrictTokenLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # scores: [batch_size, vocab_size]
        mask = torch.full_like(scores, float('-inf'))
        allowed = torch.tensor(
            list(self.allowed_token_ids), device=scores.device)
        mask[:, allowed] = 0
        return scores + mask
    
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    def insert(self, token_ids):
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]
        node.is_end = True
        node.children[1] = TrieNode()
    def get_next_tokens(self, prefix_ids):
        node = self.root

        if len(prefix_ids) == 0:
            return list(node.children.keys())
        for tid in prefix_ids:
            if tid not in node.children:
                return [] 
            node = node.children[tid]
        return list(node.children.keys())
    
class TrieConstraintLogitsProcessor(LogitsProcessor):
    def __init__(self, trie, tokenizer):
        self.trie = trie
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        batch_size, vocab_size = scores.shape
        for i in range(batch_size):
            prefix = input_ids[i].cpu().numpy()
            index = np.where(prefix == 0)[0][-1]
            allowed_tokens = self.trie.get_next_tokens(prefix[index+1:])
            if len(allowed_tokens) == 0:
                scores[i, :] = -float("inf")
            else:
                mask = torch.full_like(scores[i], -float("inf"))
                mask[allowed_tokens] = scores[i, allowed_tokens]
                scores[i] = mask
        return scores



def evaluate(model, tokenizer, test_dataset, logits_processor, test_batch_size):
    model.eval()
    dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        collate_fn=QueryEvalCollator(
            tokenizer=tokenizer,
            padding='longest',
        ),
        shuffle=False,
        drop_last=False,
        num_workers=1,
    )

    K_LIST = [10, 20, 50]

    hits_at_k = {k: 0 for k in K_LIST}
    recall_at_k_sum = {k: 0.0 for k in K_LIST}

    for batch in tqdm(dataloader, desc='Evaluating test queries'):
        inputs, labels = batch
        with torch.no_grad():
            batch_beams = model.generate(
                inputs['input_ids'].to(model.device),
                max_new_tokens=30,
                num_beams=50,             
                logits_processor=logits_processor,
                num_return_sequences=50, 
                do_sample=False,
                early_stopping=True,
            )

            batch_beams = batch_beams.reshape(inputs['input_ids'].shape[0], -1, batch_beams.shape[-1])

        for beams, label in zip(batch_beams, labels):
            rel_codes = [str(x) for x in label]
            rel_set = set(rel_codes)
            num_rel = len(rel_set)
            if num_rel == 0:
                continue

            rank_list = tokenizer.batch_decode(
                beams,
                skip_special_tokens=True,
            )
            # rank_list: ["docid1", "docid2", ...]
            list_len = len(rank_list)

            hit_positions = [i for i, pred in enumerate(rank_list) if pred in rel_set]

            if len(hit_positions) > 0:
                first_hit = min(hit_positions)
                for K in K_LIST:
                    if first_hit < min(K, list_len):
                        hits_at_k[K] += 1

            for K in K_LIST:
                effective_k = min(K, list_len)
                topk = rank_list[:effective_k]
                rel_in_topk = sum(1 for x in topk if x in rel_set)
                recall_at_k_sum[K] += rel_in_topk / num_rel

    total = len(test_dataset)
    metrics = {}
    for K in K_LIST:
        metrics[f"Hits@{K}"] = hits_at_k[K] / total
        metrics[f"Recall@{K}"] = recall_at_k_sum[K] / total

    print("Test metrics:", metrics)
    return metrics



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='DSI')
    parser.add_argument('--LTRGR_flag', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=128) 
    parser.add_argument('--eval_batch_size', type=int, default=128) 
    parser.add_argument('--test_batch_size', type=int,
                        default=32)  
    parser.add_argument('--corpus_file', type=str, default='recall/data/item_text_codes.json') 
    parser.add_argument('--train_file', type=str, default='recall/data/query_item_code_train.json') 
    parser.add_argument('--test_file', type=str,default='recall/data/query_item_code_test.json')  
    parser.add_argument('--code_file', type=str,default='recall/data/embeddings/item_code.pt')  
    args = parser.parse_args()    


    model_name = "google/mt5-base" if not args.LTRGR_flag else f"DSI/best_model"
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    def build_trie():
        print(f"[Trie] Build trie from codes: {args.code_file}")
        codes = torch.load(args.code_file, map_location="cpu")
        if isinstance(codes, torch.Tensor):
            codes = codes.cpu().numpy()
        else:
            codes = np.array(codes)
    
        codes_str_set = set(" ".join(str(j) for j in row) for row in codes)
    
        trie_local = Trie()
        for s in tqdm(codes_str_set, desc="[Trie] insert codes"):
            token_ids = tokenizer.encode(s, add_special_tokens=False)
            trie_local.insert(token_ids)
    
        return trie_local
    
    trie = build_trie()
    TrieProcessor = [TrieConstraintLogitsProcessor(trie, tokenizer)]

    output_dir = f"{args.model}/" if not args.LTRGR_flag else "LTRGR/"
    train_mode = 'gen' if not args.LTRGR_flag else 'rank'
    L = 32

    SPIECE_UNDERLINE = "▁" 
    INT_TOKEN_IDS = []
    for token, id in tokenizer.get_vocab().items():
        if token[0] == SPIECE_UNDERLINE:
            if token[1:].isdigit():
                INT_TOKEN_IDS.append(id)
        if token == SPIECE_UNDERLINE:
            INT_TOKEN_IDS.append(id)
        elif token.isdigit():
            INT_TOKEN_IDS.append(id)
    INT_TOKEN_IDS.append(tokenizer.eos_token_id)

    def restrict_decode_vocab(batch_idx, prefix_beam):
        return INT_TOKEN_IDS
    

    model = MT5ForConditionalGeneration.from_pretrained(model_name)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    train_flag = 'train_DSI' if args.model == 'DSI' else 'train'
    train_dataset = IndexingTrainDataset(path=args.train_file,
                                         corpus_path=args.corpus_file,
                                         max_length=L,
                                         tokenizer=tokenizer,
                                         flag=train_flag)

    # This eval set is really not the 'eval' set but used to report if the model can memorise (index) all training data points.
    eval_dataset = IndexingTrainDataset(path=args.train_file,
                                        corpus_path=args.corpus_file,
                                        max_length=L,
                                        tokenizer=tokenizer,
                                        flag='eval')

    # This is the actual eval set.
    test_dataset = IndexingTrainDataset(path=args.test_file,
                                        corpus_path=args.corpus_file,
                                        max_length=L,
                                        tokenizer=tokenizer,
                                        flag='test')

    logits_processor = [RestrictTokenLogitsProcessor(
        INT_TOKEN_IDS)] 

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=0.0001,
        warmup_ratio=0.06,            
        lr_scheduler_type="cosine",
    
        # weight_decay=0.01,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs = 30 if not args.LTRGR_flag else 5 ,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        dataloader_drop_last=False,  # necessary
        logging_steps=100,
        bf16=True,  # gives 0/nan loss at some point during training, seems this is a transformers bug.
        dataloader_num_workers=8,
        gradient_accumulation_steps=1,
        ddp_find_unused_parameters=False,
        report_to=["none"],
    )

    trainer = IndexingTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=IndexingCollator(
            tokenizer=tokenizer,
            padding='longest',
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        restrict_decode_vocab=restrict_decode_vocab,
        logits_processor=logits_processor,
        mode=train_mode,
    )
    trainer.train()

    if trainer.is_world_process_zero():

        best_model_dir = os.path.join(output_dir, "best_model")
        trainer.save_model(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)
    
        test_metrics = evaluate(
            model=trainer.model,
            #model=model,
            tokenizer=tokenizer,
            test_dataset=test_dataset,
            logits_processor=TrieProcessor,
            test_batch_size=args.test_batch_size,
        )
    
        with open(os.path.join(best_model_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, ensure_ascii=False, indent=2)
        print("=== Best model info ===")
        print("best_metric:", trainer.state.best_metric)
        print("best_model_checkpoint:", trainer.state.best_model_checkpoint)
if __name__ == "__main__":
    main()
