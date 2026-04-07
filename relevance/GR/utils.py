from torch.utils.data import Dataset
import torch
import json
import re
import os
from typing import Tuple, Dict, List
import torch.nn.functional as F
def is_main_process():
    return os.environ.get("RANK", "0") == "0"

def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
        
def format_instruction(instruction: str, query: str, doc: str) -> str:
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"

SYSTEM_PROMPT="Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
INSTRUCT='Given a web search query, retrieve relevant passages that answer the query'

@torch.no_grad()
def compute_p_yes(model, inputs: Dict[str, torch.Tensor], token_true_id: int, token_false_id: int) -> List[float]:
    # 官方写法：取最后一个位置 logits -> 只看 yes/no 两个 token 的 softmax
    logits = model(**inputs).logits[:, -1, :]  # [B, V]
    true_v = logits[:, token_true_id]
    false_v = logits[:, token_false_id]
    two = torch.stack([false_v, true_v], dim=1)  # [B, 2]  (no, yes)
    logp = F.log_softmax(two, dim=1)
    p_yes = logp[:, 1].exp().tolist()
    return p_yes
    
def extract_yesno(label) -> str:
    return "no" if int(label) == 0 else "yes"
        
def expand_all_linear_target_modules(model):
    ALLOWED_SUFFIXES = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    }

    found = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = name.split(".")[-1]
            if suffix in ALLOWED_SUFFIXES:
                found.add(suffix)

    return sorted(found)

def get_template_parts(tokenizer) -> Tuple[str, str]:
    """
    返回 (prefix_str, suffix_str)，中间的 user_content 由调用方拼接。
    逻辑对齐 eval 脚本：prefix 在 user 内容之前，suffix 在 user 内容之后、assistant 输出之前。
    """
    model_id = (getattr(tokenizer, "name_or_path", "") or "").lower()

    # Llama3.x (含 3/3.1/3.2) 走 llama chat template
    if "llama" in model_id and ("3" in model_id or "3." in model_id):
        prefix = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        )
        suffix = (
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    else:
        # 默认按 Qwen 模板（对齐你给的 eval_qwen3_reranker_jsonl.py）
        prefix = (
            "<|im_start|>system\n"
            f"{SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
        )
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix, suffix
        
def process_train(
    item: Dict,
    tokenizer,
    max_length: int,
    prefix_tokens: Dict[str, List[int]],
    suffix_tokens: Dict[str, List[int]],
) -> Tuple[List[int], List[int], List[int]]:
    query=item['query']
    document=item['text']
    user_content = format_instruction(INSTRUCT,query,document)
    answer=extract_yesno(item['label'])
    answer_tokens=tokenizer(answer,add_special_tokens=False)
    avail = max_length - len(prefix_tokens["input_ids"]) - len(suffix_tokens["input_ids"]) -len(answer_tokens["input_ids"]) 
    mid = tokenizer(
        user_content,
        add_special_tokens=False,
        truncation=True,
        max_length=avail,
    )
    
    input_ids = prefix_tokens["input_ids"] + mid["input_ids"] + suffix_tokens["input_ids"] + answer_tokens["input_ids"]
    attention_mask = prefix_tokens["attention_mask"] + mid["attention_mask"] + suffix_tokens["attention_mask"] + answer_tokens["attention_mask"]
    labels = [-100] * (len(prefix_tokens["input_ids"]) + len(mid["input_ids"]) + len(suffix_tokens["input_ids"])) + answer_tokens["input_ids"]

    return input_ids, attention_mask, labels
    
def process_test(
    item: Dict,
    tokenizer,
    max_length: int,
    prefix_tokens: Dict[str, List[int]],
    suffix_tokens: Dict[str, List[int]],
) -> Tuple[List[int], List[int], List[int]]:
    query=item['query']
    document=item['text']
    user_content = (
        f"<Instruct>: {INSTRUCT}\n"
        f"<Query>: {query}\n"
        f"<Document>: {document}"
    )
    avail = max_length - len(prefix_tokens["input_ids"]) - len(suffix_tokens["input_ids"])
    mid = tokenizer(
        user_content,
        add_special_tokens=False,
        truncation=True,
        max_length=avail,
    )
    
    input_ids = prefix_tokens["input_ids"] + mid["input_ids"] + suffix_tokens["input_ids"]
    attention_mask = prefix_tokens["attention_mask"] + mid["attention_mask"] + suffix_tokens["attention_mask"]

    return input_ids, attention_mask

class ChatYesNoDataset(Dataset):
    def __init__(self, file_path, tokenizer,max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.data.append(json.loads(line))
        prefix_str, suffix_str = get_template_parts(tokenizer)
        self.prefix_tokens = self.tokenizer(prefix_str, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer(suffix_str, add_special_tokens=False)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_ids, attention_mask, labels = process_train(
            item=item,
            tokenizer=self.tokenizer,
            max_length=self.max_length+1,
            prefix_tokens=self.prefix_tokens,
            suffix_tokens=self.suffix_tokens,
        )

        return {
          "input_ids": input_ids,                 
          "attention_mask": attention_mask,       
          "labels": labels,                       
        }



class ChatYesNoPredictDataset1(ChatYesNoDataset):
    def __getitem__(self, idx: int):
        item = self.data[idx]
        input_ids, attention_mask= process_test(
            item=item,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            prefix_tokens=self.prefix_tokens,
            suffix_tokens=self.suffix_tokens,
        )

        return {
          "input_ids": input_ids,                 
          "attention_mask": attention_mask                      
        },int(item["label"])


class ChatYesNoPredictDataset(ChatYesNoDataset):
    def __getitem__(self, idx: int):
        return self.data[idx]

def process_inputs(tokenizer, model, pairs: List[str], max_length: int,
                   prefix_tokens: List[int], suffix_tokens: List[int]) -> Dict[str, torch.Tensor]:
    # 先 tokenize 中间部分，再手动加 prefix/suffix，再 pad
    inputs = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=max_length - len(prefix_tokens) - len(suffix_tokens),
    )

    for i, ele in enumerate(inputs["input_ids"]):
        inputs["input_ids"][i] = prefix_tokens + ele + suffix_tokens

    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for k in inputs:
        inputs[k] = inputs[k].to(model.device)
    return inputs
        


