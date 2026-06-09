import os
import pandas as pd
from torch import nn
from typing import Any, Dict, List
from sklearn.metrics import accuracy_score, f1_score

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_handler import MetadataHandler
from src.utils.fine_tuning.general_utils import get_dataloader
from src.utils.models.model_tester import ModelTester
from src.utils.models.model_utils import cleanup_memory

logger = Logger()

class FaithfulnessEvaluator:
    def __init__(self, experiment_id: str, experiment_ft_dir: str, xai_algorithm: str, xai_entry: str, faith_entry: str, mask_rates: List[float], mask_rule: str):
        self.experiment_id = experiment_id
        self.experiment_ft_dir = experiment_ft_dir
        self.xai_algorithm = xai_algorithm
        self.xai_entry = xai_entry
        self.faith_entry = faith_entry
        self.mask_rates = [0.0] + mask_rates
        self.mask_rule = mask_rule
        
        self._initialize_faith_metadata()
    
    def _initialize_faith_metadata(self):
        faith_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "faithfulness", self.xai_algorithm, self.xai_entry, self.faith_entry)
        self.faith_page_level_metadata_path = os.path.join(faith_dir, "faithfulness_page_level.json")
        self.faith_crop_level_metadata_path = os.path.join(faith_dir, "faithfulness_crop_level.json")
        self.faith_logits_report_path = os.path.join(faith_dir, "faithfulness_logits_report.csv")
        self.faith_probs_report_path  = os.path.join(faith_dir, "faithfulness_probs_report.csv")
        
        try: self.faith_page_level_metadata = MetadataHandler(self.faith_page_level_metadata_path).load_metadata()
        except Exception as e: self.faith_page_level_metadata = {}
        
        try: self.faith_crop_level_metadata = MetadataHandler(self.faith_crop_level_metadata_path).load_metadata()
        except Exception as e: self.faith_crop_level_metadata = {}

        try: self._logits_report = pd.read_csv(self.faith_logits_report_path, header=0)
        except Exception as e: self._logits_report = pd.DataFrame()

        try: self._probs_report = pd.read_csv(self.faith_probs_report_path, header=0)
        except Exception as e: self._probs_report = pd.DataFrame()
    
    def __call__(self, model: nn.Module, classes: List[str], mean_: List[float], std_: List[float], exp_metadata: Dict[str, Any], ft_metadata: Dict[str, Any], device: str):
        logger.info(f"*** BEGINNING OF FAITHFULNESS EVALUATION -> Experiment: {self.experiment_id} | XAI Algorithm: {self.xai_algorithm} | XAI Entry: {self.xai_entry} | Faithfulness Entry: {self.faith_entry} ***")
        
        cl_accuracies, cl_macrof1s, pl_accuracies, pl_macrof1s = [], [], [], []
        batch_size, crop_size = ft_metadata["HYPERPARAMETERS"]["batch_size"], ft_metadata["HYPERPARAMETERS"]["crop_size"]
        sorted_classes = sorted(classes)
        report_frames: List[pd.DataFrame] = []
        probs_frames:  List[pd.DataFrame] = []

        for mr in self.mask_rates:
            already_evaluated  = str(mr) in self.faith_crop_level_metadata and str(mr) in self.faith_page_level_metadata
            already_in_report  = (not self._logits_report.empty) and (mr in self._logits_report["mask_rate"].values)

            if already_evaluated and already_in_report:
                logger.warning(f"Faithfulness for m_rate {mr} already exists in metadata. Skipping computation for this mask rate.")
                cl_accuracies.append(self.faith_crop_level_metadata[str(mr)]["accuracy"])
                cl_macrof1s.append(self.faith_crop_level_metadata[str(mr)]["macro_f1"])
                pl_accuracies.append(self.faith_page_level_metadata[str(mr)]["accuracy"])
                pl_macrof1s.append(self.faith_page_level_metadata[str(mr)]["macro_f1"])
                report_frames.append(self._logits_report[self._logits_report["mask_rate"] == mr])
                probs_frames.append(self._probs_report[self._probs_report["mask_rate"] == mr])

            else:
                logger.info(f"Evaluating faithfulness for mask rate: {mr}")
                current_test_set_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "faithfulness", self.xai_algorithm, self.xai_entry, self.faith_entry, "test_sets", str(mr))
                test_dataloader = get_dataloader(current_test_set_dir, classes, "test", batch_size, model.get_input_size(), mean_, std_, device)
                dataset, test_dl = test_dataloader.load_data()

                model_tester = ModelTester(self.experiment_ft_dir, model, test_dl, device, ft_metadata, exp_metadata)
                cl_labels, cl_preds, cl_logits, cl_probs, crop_preds_per_page, pl_labels, pl_preds = model_tester()

                cl_accuracy, pl_accuracy = accuracy_score(cl_labels, cl_preds), accuracy_score(pl_labels, pl_preds)
                cl_macrof1, pl_macrof1 = f1_score(cl_labels, cl_preds, average="macro"), f1_score(pl_labels, pl_preds, average="macro")

                # Build logit and probs report rows for this mask rate
                rows, prob_rows = [], []
                for (path, _), logit_vec, prob_vec in zip(dataset.samples, cl_logits, cl_probs):
                    crop_filename = os.path.basename(path)
                    true_class    = os.path.basename(os.path.dirname(path))
                    instance_name = crop_filename.rsplit("_crop", 1)[0]
                    base = {"mask_rate": mr, "instance_name": instance_name, "crop_filename": crop_filename, "true_class": true_class}
                    row      = {**base, **{f"logit_{cls}": v for cls, v in zip(sorted_classes, logit_vec)}}
                    prob_row = {**base, **{f"prob_{cls}": v  for cls, v in zip(sorted_classes, prob_vec)}}
                    rows.append(row)
                    prob_rows.append(prob_row)
                report_frames.append(pd.DataFrame(rows))
                probs_frames.append(pd.DataFrame(prob_rows))

                self.faith_crop_level_metadata[str(mr)] = {"accuracy": cl_accuracy, "macro_f1": cl_macrof1}
                self.faith_page_level_metadata[str(mr)] = {"accuracy": pl_accuracy, "macro_f1": pl_macrof1}

                cl_accuracies.append(cl_accuracy)
                cl_macrof1s.append(cl_macrof1)
                pl_accuracies.append(pl_accuracy)
                pl_macrof1s.append(pl_macrof1)

                MetadataHandler(self.faith_crop_level_metadata_path).save_metadata(self.faith_crop_level_metadata)
                MetadataHandler(self.faith_page_level_metadata_path).save_metadata(self.faith_page_level_metadata)
                del dataset, test_dl, test_dataloader, model_tester
                cleanup_memory(device)

        full_report  = pd.concat(report_frames, ignore_index=True) if report_frames else pd.DataFrame()
        full_probs   = pd.concat(probs_frames,  ignore_index=True) if probs_frames  else pd.DataFrame()
        full_report.to_csv(self.faith_logits_report_path, index=False)
        full_probs.to_csv(self.faith_probs_report_path,   index=False)
        return full_report, full_probs
