from dataclasses import dataclass
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, DataCollatorWithPadding
import json

def safe_get(doc: dict, key: str):
    val = doc.get(key, "")
    if val is None:
        return ""
    return str(val)


def build_passage_text(doc: dict):

    text = safe_get(doc, "item_title")

    return text
    
class Item2QueryTrainDataset(Dataset):
    def __init__(
            self,
            path_to_data,
            max_length: int,
            tokenizer: PreTrainedTokenizer,
    ):

        self.train_data = []
        with open(path_to_data, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                self.train_data.append(obj)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.train_data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        data = self.train_data[item]
        
        input_ids = self.tokenizer(data['item'],
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=self.max_length).input_ids[0]
        return input_ids, str(data['query'])
        
class GenerateDataset(Dataset):

    def __init__(
            self,
            path_to_data,
            max_length: int,
            tokenizer: PreTrainedTokenizer,
    ):
        self.data = []
        with open(path_to_data, "r", encoding="utf-8") as f:
            for line in f:

                obj = json.loads(line)
                item_id=obj.get("item_id")
                self.data.append((item_id, f'为下面的商品标题生成一个用户搜索query：{build_passage_text(obj)}'))


        self.max_length = max_length
        self.tokenizer = tokenizer
        self.total_len = len(self.data)


    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        item_id, text = self.data[item]
        input_ids = self.tokenizer(text,
                                   return_tensors="pt",
                                   truncation=True,
                                   max_length=self.max_length).input_ids[0]
        return input_ids, int(item_id)
        
@dataclass
class Item2QueryCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        queries = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        labels = self.tokenizer(
            queries, padding="longest", return_tensors="pt"
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels
        return inputs
        
@dataclass
class QueryEvalCollator(DataCollatorWithPadding):
    def __call__(self, features):
        input_ids = [{'input_ids': x[0]} for x in features]
        labels = [x[1] for x in features]
        inputs = super().__call__(input_ids)

        return inputs, labels


