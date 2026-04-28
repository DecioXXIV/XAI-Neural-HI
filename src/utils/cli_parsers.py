import sys
import numpy as np
from argparse import ArgumentParser
from typing import List, Tuple

from src import CLASSIFIERS, METRICS, OPTIMIZERS, LR_SCHEDULERS, FT_MODES
from data import DATASETS, SCRIBES_TO_DATASET
from src.utils.logger import Logger

logger = Logger()

### ################ ###
### SETUP EXPERIMENT ###
### ################ ###
def get_setup_exp_args():
    parser = ArgumentParser()
    parser.add_argument("-experiment_id", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True, choices=CLASSIFIERS)
    parser.add_argument("-dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("-classes", type=str, required=True)
    return _validate_setup_args(parser.parse_args())

def _validate_setup_args(args) -> Tuple[str, str, str, List[str]]:
    experiment_id, model_name, dataset, classes = args.experiment_id, args.model_name, args.dataset, args.classes
    
    classes = classes.split(',')
    valid_classes = SCRIBES_TO_DATASET.get(dataset)
    for cls in classes:
        if cls not in valid_classes:
            logger.critical(f"Invalid class '{cls}' for dataset '{dataset}'. Valid classes are: {valid_classes}")
            sys.exit()

    return experiment_id, model_name, dataset, classes

### ########### ###
### FINE TUNING ###
### ########### ###
def get_ft_args():
    parser = ArgumentParser()
    parser.add_argument("-experiment_id", type=str, required=True)
    parser.add_argument("-steering_metric", type=str, default="loss", choices=METRICS)
    parser.add_argument("-ch_layers", type=str, required=True)
    parser.add_argument("-crop_size", type=int, required=True)
    parser.add_argument("-batch_size", type=int, required=True)
    parser.add_argument("-opt", type=str, required=True, choices=OPTIMIZERS)
    parser.add_argument("-weight_decay", type=float, default=0.0001)
    parser.add_argument("-lr", type=float, required=True)
    parser.add_argument("-lr_scheduler", type=str, required=True, choices=LR_SCHEDULERS)
    parser.add_argument("-lr_final_decay_ratio", type=float, default=0.01)
    parser.add_argument("-label_smoothing", type=float, default=0.0)
    parser.add_argument("-early_stopping", type=str, default="true")
    parser.add_argument("-train_replicas", type=int, default=1)
    parser.add_argument("-random_seed", type=int, default=None)
    parser.add_argument("-epochs", type=int, default=50)
    parser.add_argument("-ft_mode", type=str, required=True, choices=FT_MODES)
    parser.add_argument("-keep_crops", type=str, default="false")
    return _validate_ft_args(parser.parse_args())

def _validate_ft_args(args):
    experiment_id = args.experiment_id
    metric, ch_layers, batch_size, crop_size = args.steering_metric, args.ch_layers, args.batch_size, args.crop_size
    opt, lr, lr_scheduler, lr_final_decay_ratio, weight_decay, label_smoothing = args.opt, args.lr, args.lr_scheduler, args.lr_final_decay_ratio, args.weight_decay, args.label_smoothing
    early_stopping, train_replicas, random_seed, epochs, ft_mode, keep_crops = args.early_stopping, args.train_replicas, args.random_seed, args.epochs, args.ft_mode, args.keep_crops
    
    error_trigger = False
    if crop_size <= 0: 
        logger.critical("crop_size must be a positive integer")
        error_trigger = True
    if batch_size <= 0: 
        logger.critical("batch_size must be a positive integer")
        error_trigger = True
    if train_replicas <= 0: 
        logger.critical("train_replicas must be a positive integer")
        error_trigger = True
    if lr <= 0: 
        logger.critical("lr must be a positive float")
        error_trigger = True
    if not (0 < lr_final_decay_ratio < 1):
        logger.critical("lr_final_decay_ratio must be a float in the range (0, 1)")
        error_trigger = True
    if not (0 <= weight_decay < 1):
        logger.critical("weight_decay must be a non-negative float less than 1")
        error_trigger = True
    if not (0 <= label_smoothing < 1):
        logger.critical("label_smoothing must be a non-negative float less than 1")
        error_trigger = True
    if epochs <= 0: 
        logger.critical("epochs must be a positive integer")
        error_trigger = True
    if random_seed is not None and random_seed < 0:
        logger.critical("random_seed must be a non-negative integer")
        error_trigger = True
    
    early_stopping, keep_crops = args.early_stopping.lower(), args.keep_crops.lower()
    if early_stopping not in ["true", "false"]:
        logger.critical("early_stopping must be 'true' or 'false'")
        error_trigger = True
    if keep_crops not in ["true", "false"]:
        logger.critical("keep_crops must be 'true' or 'false'")
        error_trigger = True
    
    if error_trigger: sys.exit()
    
    if random_seed is None:
        random_seed = int(np.random.randint(0, 2**32 - 1))
        logger.warning(f"No random_seed provided. Using '{random_seed}' (randomly generated).")

    return experiment_id, metric, ch_layers, crop_size, batch_size, opt, lr, lr_scheduler, lr_final_decay_ratio, weight_decay, label_smoothing, early_stopping, train_replicas, random_seed, epochs, ft_mode, keep_crops