from dataclasses import dataclass

import pandas as pd
import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
from tqdm import tqdm
import pandas as pd
import numpy as np

class IndexingTrainDataset(Dataset):
    def __init__(
            self,
            path,
            corpus_path,
            max_length: int,
            tokenizer: PreTrainedTokenizer,
            flag
    ):
        self.title2id=[]
        with open(path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            print(self.data[:5])
        if flag == 'train_DSI':
            with open(corpus_path) as f:
                self.title2id = json.load(f)
            print(self.data[-5:])
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.flag=flag
        
        if flag != 'test':
            n_query = len(self.data)
            all_indices = np.arange(n_query)
            rng = np.random.RandomState(42)
            rng.shuffle(all_indices)
            
            eval_size = max(1, int(n_query * 0.01))
            eval_indices = all_indices[:eval_size]
            train_query_indices = all_indices[eval_size:]
            if flag == "eval":
                self.data = [self.data[i] for i in eval_indices]
            else:
                # train / train_DSI：item2id + 99% query2id
                train_query = [self.data[i] for i in train_query_indices]
                self.data = self.title2id + train_query
        #else:
         #   self.data=self.data[:3]
        print(flag,len(self.data))
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        data = self.data[item]
        input_ids = self.tokenizer(data['text'],
                                return_tensors="pt",
                                truncation='only_first',
                                max_length=self.max_length).input_ids[0]
        return input_ids, data['output']


@dataclass
class IndexingCollator(DataCollatorWithPadding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        docids = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            docids, padding="longest", return_tensors="pt"
        ).input_ids

        # replace padding token id's of the labels by -100 according to https://huggingface.co/docs/transformers/model_doc/t5#training
        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs


@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, features):

        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels