import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig
)
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from dataset import SentencePairDataset, SentencePairPredictDataset
from parameters import load_parameters
import transformers
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import softmax
from transformers import set_seed
set_seed(42)
transformers.logging.set_verbosity_error()
os.environ['WANDB_MODE'] = 'offline'



# Configuration
model_args, data_args, training_args = load_parameters()
    
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load config
config_peft = None
if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) and 'adapter_config.json' in os.listdir(model_args.model_name_or_path):
    config_peft = PeftConfig.from_pretrained(
        model_args.model_name_or_path
    )
    config = AutoConfig.from_pretrained(
        config_peft.base_model_name_or_path,
        num_labels=2
    )
else:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2
    )

tokenizer = AutoTokenizer.from_pretrained(
    config_peft.base_model_name_or_path if isinstance(config_peft, PeftConfig) else model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer
)
if not hasattr(tokenizer, 'pad_token'):
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    cache_dir=model_args.cache,
    num_labels=2,
    problem_type="single_label_classification"
)
if model_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)


sentence1_str = 'query'
sentence2_str = 'text'

train_dataset = SentencePairDataset(data_args.train_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
eval_dataset = SentencePairDataset(data_args.validation_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
test_dataset = SentencePairPredictDataset(data_args.test_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    data_collator = data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# Train the model
print("Starting training...")
train_result = trainer.train()

trainer.save_model()
tokenizer.save_pretrained(training_args.output_dir)

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)
all_preds = []
all_labels = []
all_pos_scores = []


for i, batch in enumerate(tqdm(dataloader)):
    # Tokenize inputs
    inputs = tokenizer(
        batch[sentence1_str], batch[sentence2_str],
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=data_args.max_seq_length
        ).to(model.device)

        # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # [B, 2]
    probs = torch.softmax(logits, dim=-1)  # [B, 2]
    pos_scores = probs[:, 1]               # [B]

    # Get prediction (0 or 1)
    prediction = torch.argmax(outputs.logits, dim=1).tolist()
    all_preds.extend(prediction)
    all_labels.extend(batch["label"].cpu().tolist())
    all_pos_scores.extend(pos_scores.detach().cpu().tolist())
            
# AUC
roc_auc = roc_auc_score(all_labels, all_pos_scores)
pr_auc = average_precision_score(all_labels, all_pos_scores)

acc = accuracy_score(all_labels, all_preds) 
macro_f1 = f1_score(all_labels, all_preds, average="macro") 
weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
print("========== Evaluation ==========")
print(f"Accuracy     : {acc:.6f}")
print(f"Macro F1     : {macro_f1:.6f}")
print(f"Weighted F1  : {weighted_f1:.6f}")
print(f"ROC-AUC      : {roc_auc:.6f}")
print(f"PR-AUC       : {pr_auc:.6f}")
