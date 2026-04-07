import sys
from dataclasses import dataclass, field
from typing import Optional
import os

from transformers import (
    HfArgumentParser,
    TrainingArguments
)

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    train_file: Optional[str] = field(
        default=None, metadata={"help": "file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "file containing the data to predict."})





@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    use_lora: bool = field(
        default=False,
        metadata={
            "help": "whether to use peft for the model"
        },
    )
    lora_target_modules: str = field(
        default=None,
        metadata={"help": "target_modules for Lora or AdaLora (str with ',' separators)"},
    )
    lora_modules_to_save: str = field(
        default=None,
        metadata={"help": "modules_to_save for Lora or AdaLora (str with ',' separators)"},
    )


    def __post_init__(self):
        if self.lora_target_modules:
            self.lora_target_modules = self.lora_target_modules.split(',')
        if self.lora_modules_to_save:
            self.lora_modules_to_save = self.lora_modules_to_save.split(',')


def load_parameters():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if sys.argv[-1].endswith(".json"):

        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        
    training_args.load_best_model_at_end = True
    training_args.bf16 = True
    training_args.fp16 = False
    training_args.ddp_find_unused_parameters = False
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    
    return model_args, data_args, training_args