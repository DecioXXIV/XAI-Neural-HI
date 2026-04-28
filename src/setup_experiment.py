import os

from cli.arg_parsers import get_setup_exp_args
from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.metadata.metadata_utils import create_exp_metadata

if __name__ == "__main__":
    EXPERIMENT_ID, MODEL_NAME, DATASET, CLASSES = get_setup_exp_args()
    create_exp_metadata(EXPERIMENT_ID, MODEL_NAME, DATASET, CLASSES)
    
    EXPERIMENT_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID)
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)