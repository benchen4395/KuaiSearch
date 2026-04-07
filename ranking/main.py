import argparse
from datasets import TrainingDataProcessor
from models import *
from trainer import DCNAccelerateTrainer
import os
import random
import numpy as np
import torch
import os
def set_global_seed(seed: int):
    # Python
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN (reproducible but slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Fix seed for DataLoader workers (important)
    os.environ["PYTHONHASHSEED"] = str(seed)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir", type=str, default='data_all/dcn'
    )
    
    parser.add_argument(
        "--model", type=str, default='DCNv2',
        choices=['DCNv1', 'DCNv2', 'DNN', 'WideDeep', 'DIN'],
        help='Model to use for training'
    )

    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument(
        "--mixed_precision", type=str, default="bf16",
        choices=["no", "fp16", "bf16"]
    )

    parser.add_argument("--save_dir", type=str, default="./checkpoints_widedeep")

    args = parser.parse_args()
    set_global_seed(42)
    # 1. Data preprocessing and splitting
    print("-----Loading data-------")
    data_proc = TrainingDataProcessor(
        dataset_name_or_path=args.data_dir,
        batch_size=args.batch_size,
        max_history_len=20,
        valid_ratio=0.1,   # Validation set = 1% of training set
        seed=42,
    )

    train_loader = data_proc.get_train_dataloader()
    valid_loader = data_proc.get_valid_dataloader()
    test_loader  = data_proc.get_test_dataloader()

    # 2. Build model
    model_params = {
        'config': {},
        'num_cross_layers': 3,
        'hidden_size': 256,
        'dropout_rate': 0.3,
        'user_id_embedding_dim': 32,
    }
    
    if args.model == 'DCNv1':
        model = DCNModel(**model_params, version='v1')
    elif args.model == 'DCNv2':
        model = DCNModel(**model_params, version='v2')
    elif args.model == 'DNN':
        model = DNNModel(**model_params)
    elif args.model == 'WideDeep':
        model = WideDeepModel(**model_params)
    elif args.model == 'DIN':
        model = DINModel(**model_params)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # 3. Initialize Trainer
    print("-----Start training-------")
    trainer = DCNAccelerateTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        early_stop_patience=2,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=1,
        max_grad_norm=5.0,
        log_with=None,
        save_dir=args.save_dir,
    )

    # 4. Training (with early stopping + save best)
    trainer.train()

    # 5. Evaluate best model on test set
    trainer.load_checkpoint()

    
    test_loss, test_auc = trainer.evaluate_test()

    if trainer.is_main:
        trainer.accelerator.print(
            f"[TEST RESULT] LogLoss={test_loss:.6f}, AUC={test_auc:.6f}"
        )


if __name__ == "__main__":
    main()
