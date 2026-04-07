import sys
from dataclasses import dataclass, field
from typing import Optional
import os

from sentence_transformers import SentenceTransformerTrainingArguments
from transformers import HfArgumentParser

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
    outputs: Optional[str] = field(
        default="submit.json", metadata={"help": "The output file name for predictions"}
    )




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )



def load_parameters():
    print(sys.argv)
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, SentenceTransformerTrainingArguments))
    if sys.argv[-1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.load_best_model_at_end = True
    return model_args, data_args, training_args