import os
import time
from typing import Optional, Any, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from accelerate import Accelerator
from accelerate.utils import tqdm

class DCNAccelerateTrainer:
    """
    DCN Trainer using HuggingFace Accelerate.

    Features:
    - Single GPU / Multi-GPU (automatically handled by accelerate)
    - Support fp16 / bf16 / fp32 mixed precision
    - Loss unified through model.get_loss(...)
    - Global AUC (using accelerator.gather_for_metrics for aggregation)
    - Gradient clipping, Early Stopping, best checkpoint saving
    - ✅ Use validation loss as early stopping and LR scheduling metric
    - ✅ Support test set logloss & AUC evaluation (evaluate_test)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        num_epochs: int = 10,
        early_stop_patience: int = 2,
        save_dir: str = "./checkpoints",
        mixed_precision: str = "bf16",       # "no" / "fp16" / "bf16"
        gradient_accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = 5.0,
        log_with: Optional[str] = None,      # e.g. "tensorboard", "wandb"
    ):
        self.num_epochs = num_epochs
        self.early_stop_patience = early_stop_patience
        self.max_grad_norm = max_grad_norm
        self.criterion = nn.BCEWithLogitsLoss()
        # Initialize Accelerator (automatically handles device / DDP / AMP / grad_accum)
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=log_with,
        )
        self.is_main = self.accelerator.is_main_process

        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.best_ckpt_path = os.path.join(self.save_dir, "best_model.pt")

        # ✅ Use loss as primary metric
        self.best_valid_loss = float("inf")
        # Still keep best_valid_auc for logging only, not used for early stop / scheduling
        self.best_valid_auc = 0.0

        self.model = model

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # ✅ Adjust learning rate with validation loss (mode="min")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            verbose=self.is_main,
        )

        # Pass model / optimizer / dataloader to accelerator
        # Note: test_loader is also prepared for later use in evaluate_test
        if valid_loader is not None:
            if test_loader is not None:
                self.model, self.optimizer, self.train_loader, self.valid_loader, self.test_loader = \
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        train_loader,
                        valid_loader,
                        test_loader,
                    )
            else:
                self.model, self.optimizer, self.train_loader, self.valid_loader = \
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        train_loader,
                        valid_loader,
                    )
                self.test_loader = None
        else:
            if test_loader is not None:
                self.model, self.optimizer, self.train_loader, self.test_loader = \
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        train_loader,
                        test_loader,
                    )
                self.valid_loader = None
            else:
                self.model, self.optimizer, self.train_loader = \
                    self.accelerator.prepare(
                        self.model,
                        self.optimizer,
                        train_loader,
                    )
                self.valid_loader = None
                self.test_loader = None

    # -----------------------------
    # Unpack 4 parts from batch
    # -----------------------------
    def _unpack_batch(self, batch: Any):
        if isinstance(batch, (list, tuple)) and len(batch) == 4:
            query_features, user_features, item_features, labels = batch
        elif isinstance(batch, dict):
            query_features = batch["query_features"]
            user_features = batch["user_features"]
            item_features = batch["item_features"]
            labels = batch["labels"]
        else:
            raise ValueError("Unexpected batch format, please adapt trainer to your DataLoader.")
        return query_features, user_features, item_features, labels

    # -----------------------------
    # Train one epoch
    # -----------------------------
    def _train_one_epoch(self, epoch: int) -> Tuple[float, Optional[float]]:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        all_preds_list = []
        all_labels_list = []

        start_time = time.time()
        progress_bar = tqdm(
            self.train_loader,
            disable=not self.is_main,
            desc=f"Epoch {epoch} [train]"
        )
        for step, batch in enumerate(progress_bar):
            query_features, user_features, item_features, labels = self._unpack_batch(batch)
            labels = labels.float()

            batch_size = labels.size(0)
            total_samples += batch_size

            with self.accelerator.accumulate(self.model):
                # Use model's internal loss function
                logits = self.model(query_features, user_features, item_features)
                loss = self.criterion(logits, labels)

                # Backward
                self.accelerator.backward(loss)

                if self.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * batch_size
            if self.is_main:
                avg_loss = total_loss / max(total_samples, 1)
                progress_bar.set_postfix(loss=f"{avg_loss:.6f}")

        avg_loss = total_loss / max(total_samples, 1)
        elapsed = time.time() - start_time
        if self.is_main:
            self.accelerator.print(
                f"[Epoch {epoch}] Train Loss: {avg_loss:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

        return avg_loss,avg_loss

    # -----------------------------
    # Validation
    # -----------------------------
    @torch.no_grad()
    def _evaluate(self, epoch: int) -> Tuple[Optional[float], Optional[float]]:
        if self.valid_loader is None:
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds_list = []
        all_labels_list = []

        for batch in self.valid_loader:
            query_features, user_features, item_features, labels = self._unpack_batch(batch)
            labels = labels.float()

            # Use model's get_loss for validation
            logits = self.model(query_features, user_features, item_features)
            loss = self.criterion(logits, labels)

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            probs = torch.sigmoid(logits)
            gathered_probs = self.accelerator.gather_for_metrics(probs)
            gathered_labels = self.accelerator.gather_for_metrics(labels)

            all_preds_list.append(gathered_probs.cpu().numpy())
            all_labels_list.append(gathered_labels.cpu().numpy())

        if len(all_preds_list) > 0:
            all_preds = np.concatenate(all_preds_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
        else:
            all_preds = np.array([])
            all_labels = np.array([])

        val_loss = total_loss / max(total_samples, 1) if total_samples > 0 else None
        val_auc = None

        if self.is_main and all_preds.size > 0:
            try:
                val_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                val_auc = 0.0

            self.accelerator.print(
                f"[Epoch {epoch}] Valid Loss: {val_loss:.6f} | "
                f"Valid AUC: {val_auc:.6f}"
            )

        return val_loss, val_auc

    # -----------------------------
    # Test set evaluation: logloss & AUC
    # -----------------------------
    @torch.no_grad()
    def evaluate_test(self) -> Tuple[Optional[float], Optional[float]]:
        """
        Evaluate on test_loader:
        - test_logloss: average BCE loss
        - test_auc: ROC AUC
        Does not affect training state, no early stop / scheduling.
        """
        if self.test_loader is None:
            if self.is_main:
                self.accelerator.print("[WARN] No test_loader provided, skip test evaluation.")
            return None, None

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        all_preds_list = []
        all_labels_list = []

        for batch in self.test_loader:
            query_features, user_features, item_features, labels = self._unpack_batch(batch)
            labels = labels.float()
            
            logits = self.model(query_features, user_features, item_features)
            loss = self.criterion(logits, labels)
            # Use model's get_loss

            batch_size = labels.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size


            probs = torch.sigmoid(logits)
            gathered_probs = self.accelerator.gather_for_metrics(probs)
            gathered_labels = self.accelerator.gather_for_metrics(labels)

            all_preds_list.append(gathered_probs.cpu().numpy())
            all_labels_list.append(gathered_labels.cpu().numpy())

        if len(all_preds_list) > 0:
            all_preds = np.concatenate(all_preds_list, axis=0)
            all_labels = np.concatenate(all_labels_list, axis=0)
        else:
            all_preds = np.array([])
            all_labels = np.array([])

        test_loss = total_loss / max(total_samples, 1) if total_samples > 0 else None
        test_auc = None

        if self.is_main and all_preds.size > 0:
            try:
                test_auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                test_auc = 0.0

            self.accelerator.print(
                f"[Test] LogLoss: {test_loss:.6f} | AUC: {test_auc:.6f}"
            )

        return test_loss, test_auc

    # -----------------------------
    # Training Entry Point
    # -----------------------------
    def train(self):
        no_improve_epochs = 0

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_auc = self._train_one_epoch(epoch)

            # No validation set, just save the latest model
            if self.valid_loader is None:
                if self.is_main:
                    self._save_checkpoint(self.best_ckpt_path)
                    
                continue

            val_loss, val_auc = self._evaluate(epoch)
            test_loss, test_auc = self.evaluate_test()

            # ✅ Adjust lr: use val_loss as metric (lower is better)
            if self.is_main and self.scheduler is not None and val_loss is not None:
                self.scheduler.step(val_loss)

            # ✅ Early stopping + best ckpt: use val_loss as metric
            stop_flag = False
            if self.is_main and val_loss is not None:
                if val_loss < self.best_valid_loss:
                    self.accelerator.print(
                        f"[Epoch {epoch}] New best Loss: {self.best_valid_loss:.6f} -> {val_loss:.6f}. "
                        f"Saving checkpoint to {self.best_ckpt_path}"
                    )
                    self.best_valid_loss = val_loss
                    # Still log current val_auc
                    if val_auc is not None:
                        self.best_valid_auc = val_auc
                    self._save_checkpoint(self.best_ckpt_path)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1
                    self.accelerator.print(
                        f"[Epoch {epoch}] No improvement (val_loss={val_loss:.6f}) "
                        f"for {no_improve_epochs} epoch(s)."
                    )

                if self.early_stop_patience is not None and no_improve_epochs >= self.early_stop_patience:
                    stop_flag = True

            # Broadcast stop_flag to all processes
            stop_tensor = torch.tensor(int(stop_flag), device=self.accelerator.device)
            # All processes will get the same reduced value
            stop_tensor = self.accelerator.reduce(stop_tensor, reduction="sum")
            stop_flag = bool(stop_tensor.item())
        
            if stop_flag:
                if self.is_main:
                    self.accelerator.print(f"Early stopping triggered at epoch {epoch}.")
                break
                

    # -----------------------------
    # Save / Load
    # -----------------------------
    def _save_checkpoint(self, path: str):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        state_dict = unwrapped_model.state_dict()
        self.accelerator.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_loss": self.best_valid_loss,
                "best_valid_auc": self.best_valid_auc,
            },
            path,
        )

    def load_checkpoint(self, path: Optional[str] = None):
        if path is None:
            path = self.best_ckpt_path
        if not os.path.exists(path):
            if self.is_main:
                self.accelerator.print(f"[WARN] checkpoint {path} not found, skip loading.")
            return

        map_location = self.accelerator.device
        ckpt = torch.load(path, map_location=map_location)

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        # Compatible with old ckpt (might not have best_valid_loss)
        self.best_valid_loss = ckpt.get("best_valid_loss", float("inf"))
        self.best_valid_auc = ckpt.get("best_valid_auc", 0.0)

        if self.is_main:
            self.accelerator.print(
                f"[INFO] Loaded checkpoint from {path}, "
                f"best_valid_loss = {self.best_valid_loss:.6f}, "
                f"best_valid_auc = {self.best_valid_auc:.6f}"
            )
