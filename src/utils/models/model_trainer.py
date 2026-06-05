import os, sys, torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

from src.utils.logger import Logger
from src.utils.fine_tuning.checkpoint_saver import CheckpointSaver
from src.utils.fine_tuning.history_handler import HistoryHandler
from src.utils.fine_tuning.early_stopping import EarlyStopping
from src.utils.metadata.metadata_handler import MetadataHandler

from src.utils.data.dataloaders import TrainDataLoader, TestDataLoader

logger = Logger()

class ModelTrainer:
    def __init__(self, base_dir: str, model: nn.Module, train_dl: TrainDataLoader, val_dl: TestDataLoader, device: str, ft_metadata: Dict[str, Any], last_cp: Dict[str, Any] | None = None):
        self.base_dir = base_dir
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.ft_metadata = ft_metadata
        self.last_cp = last_cp
        self._current_epoch: int = ft_metadata["FINE_TUNING_DETAILS"].get("EPOCHS_COMPLETED", 0) + 1

        self.metric = ft_metadata["HYPERPARAMETERS"]["metric"]
        self.optimizer_type = ft_metadata["HYPERPARAMETERS"]["optimizer"]
        self.lr = ft_metadata["HYPERPARAMETERS"]["lr"]
        self.scheduler_type = ft_metadata["HYPERPARAMETERS"]["lr_scheduler"]
        self.lr_final_decay_ratio = ft_metadata["HYPERPARAMETERS"]["lr_final_decay_ratio"]
        self.use_early_stopping = ft_metadata["HYPERPARAMETERS"]["early_stopping"]
        self.early_stopping_patience = ft_metadata["HYPERPARAMETERS"]["early_stopping_patience"]
        self.num_epochs = ft_metadata["HYPERPARAMETERS"]["total_epochs"]
        
        _, self.t_dl = self.train_dl.load_data()
        _, self.v_dl = self.val_dl.load_data()

        self.use_amp = device.startswith("cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        if self.last_cp is not None and self.last_cp.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(self.last_cp["scaler_state_dict"])

        if self.use_early_stopping:
            self.max_epochs = 100
            self.early_stopping = EarlyStopping(metric=self.metric, patience=self.early_stopping_patience, delta=0.0, max_epochs=self.max_epochs)
        else:
            self.early_stopping = None
            self.max_epochs = self.num_epochs

    def _set_optimizer(self, weight_decay: float) -> Optimizer:
        if self.optimizer_type.lower() == "sgd":
            optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, momentum=0.9, nesterov=False, weight_decay=weight_decay)
        elif self.optimizer_type.lower() == "adam":
            optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=[0.9, 0.999], weight_decay=weight_decay)
        elif self.optimizer_type.lower() == "adamw":
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=[0.9, 0.999], weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer '{self.optimizer_type}' not recognized!")

        if self.last_cp is not None:
            optimizer.load_state_dict(self.last_cp["optimizer_state_dict"])

        return optimizer

    def _set_scheduler(self, optimizer: Optimizer) -> LRScheduler | None:
        scheduler = None
        min_lr = self.lr * self.lr_final_decay_ratio
        last_epoch = self.ft_metadata["FINE_TUNING_DETAILS"].get("EPOCHS_COMPLETED", -1)

        if self.scheduler_type == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, self.num_epochs, min_lr, last_epoch)

        if scheduler is not None and self.last_cp is not None:
            scheduler.load_state_dict(self.last_cp["scheduler_state_dict"])

        return scheduler

    def _reset_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        t_max = self.max_epochs - self.num_epochs
        eta_min = self.lr * self.lr_final_decay_ratio * 0.1
        return CosineAnnealingLR(optimizer, t_max, eta_min, last_epoch=-1)

    def _compute_minibatch_accuracy(self, output: torch.Tensor, label: torch.Tensor) -> Tuple[int, torch.Tensor]:
        max_index = output.argmax(dim=1)
        correct = (max_index == label).sum().item()
        return correct, max_index

    def _train_one_epoch(self, optimizer: Optimizer, criterion: nn.Module) -> Tuple[float, float, float, float]:
        self.train_dl.update_epoch(self._current_epoch)
        self.model.train()
        epoch_loss, epoch_acc, total_samples = 0.0, 0.0, 0
        ds_output, ds_target = [], []
        pbar = tqdm(self.t_dl, desc="Training", dynamic_ncols=True)

        for data, target in pbar:
            bs = data.size(0)
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                output = self.model(data)
                loss = criterion(output, target)
            correct, max_index = self._compute_minibatch_accuracy(output, target)

            epoch_loss += loss.item() * bs
            epoch_acc += correct
            total_samples += bs
            ds_output.append(max_index.detach())
            ds_target.append(target)

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()

            pbar.set_postfix(loss=epoch_loss / total_samples, accuracy=epoch_acc / total_samples, refresh=True)

        self._current_epoch += 1
        ds_output_np = torch.cat(ds_output).cpu().numpy()
        ds_target_np = torch.cat(ds_target).cpu().numpy()
        return epoch_loss / total_samples, epoch_acc / total_samples, f1_score(ds_target_np, ds_output_np, average='macro'), f1_score(ds_target_np, ds_output_np, average='weighted')

    def _validate_one_epoch(self, criterion: nn.Module) -> Tuple[float, float, float, float]:
        self.model.eval()
        val_loss, val_acc, total_samples = 0.0, 0.0, 0
        ds_output, ds_target = [], []
        pbar = tqdm(self.v_dl, desc="Validation", dynamic_ncols=True)

        with torch.inference_mode():
            for data, target in pbar:
                bs = data.size(0)
                data, target = data.to(self.device), target.to(self.device)

                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    output = self.model(data)
                    loss = criterion(output, target)
                correct, max_index = self._compute_minibatch_accuracy(output, target)

                val_loss += loss.item() * bs
                val_acc += correct
                total_samples += bs
                ds_output.append(max_index)
                ds_target.append(target)

                pbar.set_postfix(loss=val_loss / total_samples, accuracy=val_acc / total_samples, refresh=True)

        ds_output_np = torch.cat(ds_output).cpu().numpy()
        ds_target_np = torch.cat(ds_target).cpu().numpy()
        return val_loss / total_samples, val_acc / total_samples, f1_score(ds_target_np, ds_output_np, average='macro'), f1_score(ds_target_np, ds_output_np, average='weighted')

    def __call__(self, ft_metadata_path: str):
        self.model.to(self.device)
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            self.model = nn.DataParallel(self.model)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.ft_metadata["HYPERPARAMETERS"]["label_smoothing"])
        optimizer = self._set_optimizer(weight_decay=self.ft_metadata["HYPERPARAMETERS"]["weight_decay"])
        scheduler = self._set_scheduler(optimizer)

        history_handler = HistoryHandler(self.base_dir)
        history = history_handler.load_history()

        lower_is_better = self.metric == "loss"
        default_best = np.inf if lower_is_better else -np.inf
        agg = np.min if lower_is_better else np.max
        best_metric_t = agg(history["train"][self.metric]) if history["train"][self.metric] else default_best
        best_metric_v = agg(history["val"][self.metric]) if history["val"][self.metric] else default_best

        start_epoch = self.ft_metadata["FINE_TUNING_DETAILS"].get("EPOCHS_COMPLETED", 0) + 1
        checkpoint_saver = CheckpointSaver(self.base_dir, self.metric)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Fine-tuning mode: '{self.ft_metadata['HYPERPARAMETERS']['ft_mode']}'")
        logger.info(f"Trainable parameters: {trainable_params}")
        
        if self.early_stopping is not None:
            logger.warning(f"Early Stopping enabled: Max Epochs = {self.max_epochs}")
            logger.warning(f"Validation {self.metric.capitalize()} check triggers from epoch: {self.num_epochs}")
            if start_epoch > 1: self.early_stopping.load_state_dict(self.last_cp["early_stopping"])
            self.early_stopping.set_best_val_metric(best_metric_v)

        for epoch in range(start_epoch, self.max_epochs + 1):
            logger.info(f"Epoch {epoch} / {self.max_epochs}")

            current_lr = optimizer.param_groups[0]["lr"]
            if scheduler is not None: logger.info(f"Current Learning Rate: {current_lr}")
            history["learning_rates"].append(current_lr)

            train_epoch_loss, train_epoch_acc, train_epoch_macrof1, train_epoch_weightedf1 = self._train_one_epoch(optimizer, criterion)
            logger.info(f"Epoch {epoch} -> Train Loss: {train_epoch_loss}, Train Accuracy: {train_epoch_acc}, Train Macro F1: {train_epoch_macrof1}, Train Weighted F1: {train_epoch_weightedf1}")
            val_epoch_loss, val_epoch_acc, val_epoch_macrof1, val_epoch_weightedf1 = self._validate_one_epoch(criterion)
            logger.info(f"Epoch {epoch} -> Val Loss: {val_epoch_loss}, Val Accuracy: {val_epoch_acc}, Val Macro F1: {val_epoch_macrof1}, Val Weighted F1: {val_epoch_weightedf1}")

            history["train"]["loss"].append(train_epoch_loss)
            history["train"]["accuracy"].append(train_epoch_acc)
            history["train"]["macrof1"].append(train_epoch_macrof1)
            history["train"]["weightedf1"].append(train_epoch_weightedf1)
            history["val"]["loss"].append(val_epoch_loss)
            history["val"]["accuracy"].append(val_epoch_acc)
            history["val"]["macrof1"].append(val_epoch_macrof1)
            history["val"]["weightedf1"].append(val_epoch_weightedf1)

            train_metrics = {"loss": train_epoch_loss, "accuracy": train_epoch_acc, "macrof1": train_epoch_macrof1, "weightedf1": train_epoch_weightedf1}
            val_metrics   = {"loss": val_epoch_loss,   "accuracy": val_epoch_acc,   "macrof1": val_epoch_macrof1,   "weightedf1": val_epoch_weightedf1}
            train_metric_value, val_metric_value = train_metrics[self.metric], val_metrics[self.metric]

            best_metric_t = min(best_metric_t, train_metric_value) if lower_is_better else max(best_metric_t, train_metric_value)
            update_val_best_model = False
            if lower_is_better:
                if val_metric_value < best_metric_v:
                    best_metric_v = val_metric_value
                    update_val_best_model = True
            else:
                if val_metric_value > best_metric_v:
                    best_metric_v = val_metric_value
                    update_val_best_model = True
            
            if update_val_best_model: checkpoint_saver("val_best_model", self.model, optimizer, scheduler, self.early_stopping, self.scaler)
            checkpoint_saver("last_checkpoint", self.model, optimizer, scheduler, self.early_stopping, self.scaler)
            history_handler.save_history(history)

            self.ft_metadata["FINE_TUNING_DETAILS"]["EPOCHS_COMPLETED"] = epoch
            MetadataHandler(ft_metadata_path).save_metadata(self.ft_metadata)
            
            if self.early_stopping is not None:
                if epoch <= self.num_epochs: self.early_stopping.step_before_trigger(val_metric_value)
                else:
                    if self.early_stopping.step(val_metric_value): break
            
            if scheduler is not None:
                scheduler.step()
                if epoch == self.num_epochs:
                    logger.warning(f"Epoch {epoch} completed: resetting scheduler for the second phase of training (with early stopping)...")
                    scheduler = self._reset_scheduler(optimizer)
            
            print()