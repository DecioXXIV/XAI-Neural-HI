import os, sys
from argparse import ArgumentParser
from typing import List, Tuple

from src import CLASSIFIERS
from data import DATASETS, SCRIBES_TO_DATASET
from src.utils.constants import METADATA_ROOT, EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_handler import MetadataHandler

logger = Logger()

def get_setup_exp_args():
    parser = ArgumentParser()
    parser.add_argument("-experiment_id", type=str, required=True)
    parser.add_argument("-model_name", type=str, required=True, choices=CLASSIFIERS)
    parser.add_argument("-dataset", type=str, required=True, choices=DATASETS)
    parser.add_argument("-classes", type=str, required=True)
    return _validate_args(parser.parse_args())

def _validate_args(args) -> Tuple[str, str, str, List[str]]:
    experiment_id, model_name, dataset, classes = args.experiment_id, args.model_name, args.dataset, args.classes
    
    classes = classes.split(',')
    valid_classes = SCRIBES_TO_DATASET.get(dataset)
    for cls in classes:
        if cls not in valid_classes:
            logger.critical(f"Invalid class '{cls}' for dataset '{dataset}'. Valid classes are: {valid_classes}")
            sys.exit()
    return experiment_id, model_name, dataset, classes

def create_exp_metadata(experiment_id: str, model_name: str, dataset: str, classes: List[str]):
    os.makedirs(os.path.join(METADATA_ROOT, experiment_id), exist_ok=True)
    exp_metadata_path = os.path.join(METADATA_ROOT, experiment_id, "general-metadata.json")
    
    if os.path.exists(exp_metadata_path):
        logger.warning(f"Experiment ID '{experiment_id}' already exists. Please choose a unique experiment ID.")
        sys.exit()
    
    mh = MetadataHandler(exp_metadata_path)
    exp_metadata = {"EXPERIMENT_ID": experiment_id, "MODEL_NAME": model_name, "DATASET": dataset, "CLASSES": classes}
    mh.save_metadata(exp_metadata)
    logger.info(f"Metadata successfully created for Experiment: '{experiment_id}'")

def initialize_experiment_directory(experiment_id: str):
    experiment_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)