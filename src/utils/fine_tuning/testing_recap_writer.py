import os, itertools, json
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger

logger = Logger()

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