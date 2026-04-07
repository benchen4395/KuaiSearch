import csv
from tqdm import tqdm
import os
import transformers
transformers.logging.set_verbosity_error()
from transformers import (
    BertTokenizer,
    BertModel,
    )
import torch
import numpy as np
from accelerate import PartialState
import json
def safe_get(doc: dict, key: str) -> str:
    val = doc.get(key, "")
    if val is None:
        return ""
    return str(val)

def build_passage_text(doc: dict):
    
    fields = [
        safe_get(doc, "item_title"),
    ]
    parts = [f.strip() for f in fields if f and f.strip()]
    text = " ".join(parts)

    return text

def normalize_document(document: str):
    document = document.replace("\n", " ").replace("’", "'")
    if document.startswith('"'):
        document = document[1:]
    if document.endswith('"'):
        document = document[:-1]
    return document
    
def generate_embeddings(args):
    distributed_state = PartialState()
    device = distributed_state.device


    doc_encoder = BertModel.from_pretrained(args.pretrained_model_path,add_pooling_layer=False)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    doc_encoder.eval()
    doc_encoder.to(device)



    progress_bar = tqdm(total=args.num_docs, disable=not distributed_state.is_main_process,ncols=100,desc='loading msmarco...')
    corpus = []
    with open(args.corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = str(obj["item_id"])
            text = build_passage_text(obj)
            corpus.append((doc_id, normalize_document(text)))
            progress_bar.update(1)

    with distributed_state.split_between_processes(corpus) as sharded_corpus:
        
        sharded_corpus = [sharded_corpus[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_corpus),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_corpus), disable=not distributed_state.is_main_process,ncols=100,desc='encoding corpus...')
        doc_embeddings = []
        shard_doc_ids = []  
        for data in sharded_corpus:
            ids = [x[0] for x in data]   
            passage = [x[1] for x in data]
            model_input = tokenizer(passage,max_length=128,padding='max_length',return_tensors='pt',truncation=True).to(device)
            with torch.no_grad():
                if isinstance(doc_encoder,BertModel):
                    CLS_POS = 0
                    output = doc_encoder(**model_input).last_hidden_state[:,CLS_POS,:].cpu().numpy()
                else:
                    output = doc_encoder(**model_input).pooler_output.cpu().numpy()
            doc_embeddings.append(output)
            shard_doc_ids.extend(ids)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)
        os.makedirs(args.output_dir,exist_ok=True)
        np.save(f'{args.output_dir}/corpus_shard_{distributed_state.process_index}.npy',doc_embeddings)
        np.save(f'{args.output_dir}/corpus_shard_{distributed_state.process_index}_ids.npy', np.array(shard_doc_ids))

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_path",default="data/corpus.jsonl")
    parser.add_argument("--num_docs",type=int,default=74065501)
    parser.add_argument("--encoding_batch_size",type=int,default=2048)
    parser.add_argument("--pretrained_model_path",default="./model/best/encoder")
    parser.add_argument("--output_dir",default="doc_embedding")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    generate_embeddings(args)