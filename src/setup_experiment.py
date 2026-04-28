from src.utils.setup_exp.setup_exp_utils import get_setup_exp_args, create_exp_metadata, initialize_experiment_directory

if __name__ == "__main__":
    EXPERIMENT_ID, MODEL_NAME, DATASET, CLASSES = get_setup_exp_args()
    create_exp_metadata(EXPERIMENT_ID, MODEL_NAME, DATASET, CLASSES)
    initialize_experiment_directory(EXPERIMENT_ID)