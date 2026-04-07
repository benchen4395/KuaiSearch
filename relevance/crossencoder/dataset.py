from torch.utils.data import Dataset
import torch
import json

# Custom Dataset class
class SentencePairDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, sentence1_str, sentence2_str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.sentence1_str = sentence1_str
        self.sentence2_str = sentence2_str

        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence1 = item[self.sentence1_str]
        sentence2 = item[self.sentence2_str]
        label = item['label']

        encoding = self.tokenizer(
            sentence1,
            sentence2,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(int(label), dtype=torch.long)
        }

class SentencePairPredictDataset(SentencePairDataset):
    def __getitem__(self, idx):
        return self.data[idx]