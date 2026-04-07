import json
import os
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig
)
from utils import *
import torch
from torch.utils.data import DataLoader
from parameters import load_parameters
import transformers
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from scipy.special import softmax
from transformers import set_seed

set_seed(42)
model_args, data_args, training_args = load_parameters()


    
config_peft = None
if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) and 'adapter_config.json' in os.listdir(model_args.model_name_or_path):
    config_peft = PeftConfig.from_pretrained(
        model_args.model_name_or_path,cache_dir='cache'
    )
    config = AutoConfig.from_pretrained(
        config_peft.base_model_name_or_path,cache_dir='cache'
    )
else:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,cache_dir='cache'
    )
    
tokenizer = AutoTokenizer.from_pretrained(
    config_peft.base_model_name_or_path if isinstance(config_peft, PeftConfig) else model_args.model_name_or_path,padding_side="left",cache_dir='cache')
model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, torch_dtype=torch.bfloat16,cache_dir='cache')


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if model_args.use_lora:
    if len(model_args.lora_target_modules) == 1 and model_args.lora_target_modules[0] == "all_linear":
        model_args.lora_target_modules = expand_all_linear_target_modules(model)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

train_dataset=ChatYesNoDataset(data_args.train_file, tokenizer, data_args.max_seq_length)
eval_dataset = ChatYesNoDataset(data_args.validation_file, tokenizer, data_args.max_seq_length)
test_dataset = ChatYesNoPredictDataset(data_args.test_file, tokenizer, data_args.max_seq_length)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator = data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# Train the model
if training_args.should_log:
    print("Starting training...")
train_result = trainer.train()
trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)



dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
p_yes_all = []
y_pred = []
y_true = []
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
prefix, suffix=get_template_parts(tokenizer)
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

model.eval()
tokenizer.padding_side = "left"
for i, batch in enumerate(tqdm(dataloader)):
    queries = batch["query"]
    docs = batch["text"]
    labels = batch["label"]
    pairs = [format_instruction(INSTRUCT, q, d) for q, d in zip(queries, docs)]
    inputs = process_inputs(
                tokenizer=tokenizer,
                model=model,
                pairs=pairs,
                max_length=data_args.max_seq_length,
                prefix_tokens=prefix_tokens,
                suffix_tokens=suffix_tokens,
            )
    p_yes = compute_p_yes(model, inputs, token_true_id, token_false_id)
    pred = [1 if p >= 0.5 else 0 for p in p_yes]

    y_pred.extend(pred)
    y_true.extend(labels)
    p_yes_all.extend(p_yes)
        
acc = accuracy_score(y_true, y_pred)
macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

# ROC-AUC and PR-AUC
roc_auc = roc_auc_score(y_true, p_yes_all)
pr_auc = average_precision_score(y_true, p_yes_all)

print(f"Accuracy:   {acc:.6f}")
print(f"Macro F1:   {macro_f1:.6f}")
print(f"Weighted F1:{weighted_f1:.6f}")
print(f"ROC-AUC:    {roc_auc:.6f}")
print(f"PR-AUC:     {pr_auc:.6f}")
