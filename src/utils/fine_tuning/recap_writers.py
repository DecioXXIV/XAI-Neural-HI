import os, json, itertools
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger

logger = Logger()

class TrainingRecapWriter:
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.history_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "fine_tuning", "history")
    
    def _create_metric_recaps(self):
        metric_recap_dict = {}
        metric_recap_dict["training_infos"] = {}
        metric_recap_dict["training_infos"]["train_set"] = {}
        metric_recap_dict["training_infos"]["val_set"] = {}
        
        for metric_serie in ["losses", "accs"]:
            if metric_serie == "losses": metric = "loss"
            elif metric_serie == "accs": metric = "accuracy"
            
            values = {"train": [], "val": []}
            for phase in values.keys():
                with open(os.path.join(self.history_dir, f"{phase}_{metric_serie}.pkl"), "rb") as f:
                    values[phase] = pkl.load(f)
            
            best_train_metric, best_val_metric = None, None
            if metric_serie == "losses":
                best_train_metric = np.min(values["train"])
                best_val_metric = np.min(values["val"])
            elif metric_serie == "accs":
                best_train_metric = np.max(values["train"])
                best_val_metric = np.max(values["val"])
            best_train_epoch = np.where(np.array(values["train"]) == best_train_metric)[0][0] + 1
            best_val_epoch = np.where(np.array(values["val"]) == best_val_metric)[0][0] + 1
            
            metric_recap_dict["training_infos"]["train_set"][f"optimal_{metric}_value"] = best_train_metric, 4
            metric_recap_dict["training_infos"]["train_set"][f"epoch_optimal_{metric}_epoch"] = f"{best_train_epoch}//{len(values['train'])}"
            
            metric_recap_dict["training_infos"]["val_set"][f"optimal_{metric}_value"] = best_val_metric, 4
            metric_recap_dict["training_infos"]["val_set"][f"epoch_optimal_{metric}_epoch"] = f"{best_val_epoch}//{len(values['val'])}"
            
        with open(os.path.join(self.history_dir, "training_recap.json"), "w") as f:
            json.dump(metric_recap_dict, f, indent=4)
    
    def _plot_metric_recaps(self):
        for metric_serie in ["losses", "accs"]:
            if metric_serie == "losses": metric = "loss"
            elif metric_serie == "accs": metric = "accuracy"
            
            values = {"train": [], "val": []}
            for phase in values.keys():
                with open(os.path.join(self.history_dir, f"{phase}_{metric_serie}.pkl"), "rb") as f:
                    values[phase] = pkl.load(f)
            
            plt.plot(values['train'])
            plt.plot(values['val'])
            plt.title(f"Model {metric}")
            plt.ylabel(f"{metric} [-]")
            plt.xlabel("Epoch [-]")
            plt.legend(['Training', 'Validation'], loc='best')
            plt.savefig(os.path.join(self.history_dir, f"{metric}.png"))
            plt.close()
    
    def _plot_learning_rates(self):
        with open(os.path.join(self.history_dir, "learning_rates.pkl"), "rb") as f: lrs = pkl.load(f)
        
        plt.plot(lrs)
        plt.title("Learning Rate Schedule")
        plt.ylabel("Learning Rate [-]")
        plt.xlabel("Epoch [-]")
        plt.savefig(os.path.join(self.history_dir, "learning_rates.png"))
        plt.close()
    
    def __call__(self):
        self._create_metric_recaps()
        self._plot_metric_recaps()
        self._plot_learning_rates()

class TestingRecapWriter:
    def __init__(self, experiment_id: str, test_dl: DataLoader):
        self.experiment_id = experiment_id
        self.c_to_idx = test_dl.dataset.class_to_idx
        self.idx_to_c = {v: k for k, v in self.c_to_idx.items()}
        self.target_names = list(self.c_to_idx.keys())

        self.output_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "fine_tuning", "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_confusion_matrix(self, level: str, labels: List[int], preds: List[int]) -> None:
        label_names = np.array([self.idx_to_c[i] for i in labels])
        pred_names  = np.array([self.idx_to_c[i] for i in preds])

        cm = confusion_matrix(label_names, pred_names, labels=self.target_names)
        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        plt.figure(figsize=(20, 20))
        cmap = plt.get_cmap("Blues")
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title("Confusion matrix")
        plt.colorbar()

        tick_marks = np.arange(len(self.target_names))
        plt.xticks(tick_marks, self.target_names, rotation=45)
        plt.yticks(tick_marks, self.target_names)

        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm_norm.max() / 1.5
        for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}", horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel(f"Predicted label\naccuracy={accuracy:.4f}; misclass={misclass:.4f}")
        plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{level}.png"))
        plt.close()

    def __call__(self, crop_labels: List[int], crop_preds: List[int], page_labels: List[int], page_preds: List[int]) -> None:
        crop_metrics = classification_report(crop_labels, crop_preds, target_names=self.target_names, output_dict=True)
        self._save_confusion_matrix("crop_level", crop_labels, crop_preds)
        logger.info(f"Crop-Level Accuracy: {crop_metrics['accuracy']:.4f}")

        page_metrics = classification_report(page_labels, page_preds, target_names=self.target_names, output_dict=True)
        self._save_confusion_matrix("page_level", page_labels, page_preds)
        logger.info(f"Page-Level Accuracy: {page_metrics['accuracy']:.4f}")

        classification_metrics = {"crop_level": crop_metrics, "page_level": page_metrics}
        with open(os.path.join(self.output_dir, "classification_metrics.json"), "w") as f:
            json.dump(classification_metrics, f, indent=4)