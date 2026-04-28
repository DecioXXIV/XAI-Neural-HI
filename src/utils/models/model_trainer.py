import os, sys, torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Tuple
from tqdm import tqdm
from torch.optim import Optimizer, SGD, Adam, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.utils.data import DataLoader

from src.utils.constants import METADATA_ROOT
from src.utils.logger import Logger
from src.utils.fine_tuning.checkpoint_saver import CheckpointSaver
from src.utils.fine_tuning.history_handler import HistoryHandler
from src.utils.fine_tuning.early_stopping import EarlyStopping
from src.utils.metadata.metadata_handler import MetadataHandler

logger = Logger()


class ModelTrainer:
    def __init__(self, experiment_id: str, model: nn.Module, train_dl: DataLoader, val_dl: DataLoader, device: str, ft_metadata: Dict[str, Any], last_cp: Dict[str, Any] | None = None):
        self.experiment_id = experiment_id
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.ft_metadata = ft_metadata
        self.last_cp = last_cp

        self.optimizer_type = ft_metadata["HYPERPARAMETERS"]["optimizer"]
        self.lr = ft_metadata["HYPERPARAMETERS"]["lr"]
        self.scheduler_type = ft_metadata["HYPERPARAMETERS"]["lr_scheduler"]
        self.lr_final_decay_ratio = ft_metadata["HYPERPARAMETERS"]["lr_final_decay_ratio"]
        self.use_early_stopping = ft_metadata["HYPERPARAMETERS"]["early_stopping"]
        self.num_epochs = ft_metadata["HYPERPARAMETERS"]["total_epochs"]

        if self.use_early_stopping:
            self.early_stopping = EarlyStopping(patience=10, delta=0.0, max_epochs=200)
            self.max_epochs = self.early_stopping.max_epochs
        else:
            self.early_stopping = None
            self.max_epochs = self.num_epochs

    def _set_optimizer(self) -> Optimizer:
        if self.optimizer_type.lower() == "sgd":
            optimizer = SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, momentum=0.9, nesterov=False, weight_decay=0.0001)
        elif self.optimizer_type.lower() == "adam":
            optimizer = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=[0.9, 0.999], weight_decay=0.0001)
        elif self.optimizer_type.lower() == "adamw":
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, betas=[0.9, 0.999], weight_decay=0.01)
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

    def _reset_scheduler(self, scheduler: LRScheduler) -> LRScheduler:
        last_epoch = self.ft_metadata["FINE_TUNING_DETAILS"].get("EPOCHS_COMPLETED", 0)
        scheduler.T_max = self.max_epochs - last_epoch
        scheduler.eta_min = int(self.lr * self.lr_final_decay_ratio)
        return scheduler

    def _compute_minibatch_accuracy(self, output: torch.Tensor, label: torch.Tensor) -> Tuple[int, float]:
        max_index = output.argmax(dim=1)
        correct = (max_index == label).sum().item()
        correct_ratio = correct / label.size(0)
        return correct, correct_ratio

    def _train_one_epoch(self, optimizer: Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        epoch_loss, epoch_acc, total_samples = 0.0, 0.0, 0
        pbar = tqdm(self.train_dl, desc="Training", dynamic_ncols=True)

        for data, target in pbar:
            bs = data.size(0)
            data, target = data.to(self.device), target.to(self.device)

            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            correct, _ = self._compute_minibatch_accuracy(output, target)

            epoch_loss += loss.item() * bs
            epoch_acc += correct
            total_samples += bs

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=epoch_loss / total_samples, accuracy=epoch_acc / total_samples, refresh=True)

        return epoch_loss / total_samples, epoch_acc / total_samples

    def _validate_one_epoch(self, criterion: nn.Module) -> Tuple[float, float]:
        self.model.eval()
        val_loss, val_acc, total_samples = 0.0, 0.0, 0
        pbar = tqdm(self.val_dl, desc="Validation", dynamic_ncols=True)

        with torch.no_grad():
            for data, target in pbar:
                bs = data.size(0)
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)
                correct, _ = self._compute_minibatch_accuracy(output, target)

                val_loss += loss.item() * bs
                val_acc += correct
                total_samples += bs

                pbar.set_postfix(loss=val_loss / total_samples, accuracy=val_acc / total_samples, refresh=True)

        return val_loss / total_samples, val_acc / total_samples

    def __call__(self):
        self.model.to(self.device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = self._set_optimizer()
        scheduler = self._set_scheduler(optimizer)

        history_handler = HistoryHandler(self.experiment_id)
        train_loss = history_handler.load_history("train_losses.pkl")
        val_loss = history_handler.load_history("val_losses.pkl")
        train_acc = history_handler.load_history("train_accs.pkl")
        val_acc = history_handler.load_history("val_accs.pkl")
        learning_rates = history_handler.load_history("learning_rates.pkl")

        min_loss_t = np.min(train_loss) if train_loss else sys.maxsize
        min_loss_v = np.min(val_loss) if val_loss else sys.maxsize

        start_epoch = self.ft_metadata["FINE_TUNING_DETAILS"].get("EPOCHS_COMPLETED", 0) + 1
        checkpoint_saver = CheckpointSaver(self.experiment_id)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Fine-tuning mode: '{self.ft_metadata['HYPERPARAMETERS']['ft_mode']}'")
        logger.info(f"Trainable parameters: {trainable_params}")

        if self.use_early_stopping:
            logger.warning(f"Early Stopping enabled: Max Epochs = {self.max_epochs}")
            logger.warning(f"Validation Loss check triggers from epoch: {self.num_epochs}")
            if start_epoch > 1:
                self.early_stopping.load_state_dict(self.last_cp["early_stopping"])
            self.early_stopping.set_best_val_loss(min_loss_v)

        for epoch in range(start_epoch, self.max_epochs + 1):
            logger.info(f"Epoch {epoch} / {self.max_epochs}")

            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)

            if scheduler is not None:
                logger.info(f"Current Learning Rate: {current_lr}")
                if epoch == self.num_epochs:
                    scheduler = self._reset_scheduler(scheduler)

            train_epoch_loss, train_epoch_acc = self._train_one_epoch(optimizer, criterion)
            logger.info(f"Epoch {epoch} -> Train Loss: {train_epoch_loss}, Train Accuracy: {train_epoch_acc}")
            val_epoch_loss, val_epoch_acc = self._validate_one_epoch(criterion)
            logger.info(f"Epoch {epoch} -> Val Loss: {val_epoch_loss}, Val Accuracy: {val_epoch_acc}")

            train_loss.append(train_epoch_loss)
            train_acc.append(train_epoch_acc)
            val_loss.append(val_epoch_loss)
            val_acc.append(val_epoch_acc)

            min_loss_t = min(min_loss_t, train_epoch_loss)
            min_loss_v = checkpoint_saver(val_epoch_loss, min_loss_v, self.model, optimizer, scheduler, self.early_stopping, "val", check=True)
            checkpoint_saver(train_epoch_loss, None, self.model, optimizer, scheduler, self.early_stopping, None, check=False)

            history_handler.save_history("train_losses.pkl", train_loss)
            history_handler.save_history("val_losses.pkl", val_loss)
            history_handler.save_history("train_accs.pkl", train_acc)
            history_handler.save_history("val_accs.pkl", val_acc)
            history_handler.save_history("learning_rates.pkl", learning_rates)

            if scheduler is not None: scheduler.step()

            self.ft_metadata["FINE_TUNING_DETAILS"]["EPOCHS_COMPLETED"] = epoch
            ft_metadata_path = os.path.join(METADATA_ROOT, self.experiment_id, "ft-metadata.json")
            MetadataHandler(ft_metadata_path).save_metadata(self.ft_metadata)

            if self.use_early_stopping:
                if epoch <= self.num_epochs:
                    self.early_stopping.step_before_trigger(val_epoch_loss)
                else:
                    if self.early_stopping.step(val_epoch_loss):
                        break