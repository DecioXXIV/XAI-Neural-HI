import os, json
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

from src.utils.constants import EXPERIMENTS_ROOT

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