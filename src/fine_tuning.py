import os, torch
from datetime import datetime

from cli.arg_parsers import get_ft_args
from src.utils.constants import METADATA_ROOT, EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_experiment_metadata, get_ft_metadata, initialize_ft_metadata, add_timestamp_to_ft_metadata
from src.utils.fine_tuning.general_utils import create_dataset, get_train_rgb_mean_std, get_dataloader, remove_subdirectories
from src.utils.models.model_utils import cleanup_memory, load_model, train_model, test_model, setup_device, set_random_seed

logger = Logger()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    EXPERIMENT_ID, METRIC, CH_LAYERS, CROP_SIZE, BATCH_SIZE, OPTIMIZER, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, TRAIN_REPLICAS, RANDOM_SEED, EPOCHS, TRAIN_TRANSFORMS, FT_MODE, KEEP_CROPS = get_ft_args()
    
    EXP_METADATA = get_experiment_metadata(EXPERIMENT_ID)
    FT_METADATA = get_ft_metadata(EXPERIMENT_ID)
    FT_METADATA = initialize_ft_metadata(EXPERIMENT_ID, FT_METADATA, METRIC, CH_LAYERS, CROP_SIZE, BATCH_SIZE, OPTIMIZER, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, TRAIN_REPLICAS, RANDOM_SEED, EPOCHS, TRAIN_TRANSFORMS, FT_MODE)
    FT_METADATA_PATH = os.path.join(METADATA_ROOT, EXPERIMENT_ID, "ft-metadata.json")
    
    EXPERIMENT_FT_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning")
    os.makedirs(EXPERIMENT_FT_DIR, exist_ok=True)
    
    DEVICE = setup_device()
    
    logger.info(f"*** Experiment: {EXPERIMENT_ID} -> START OF FINE TUNING-PROCESS ***\n")
    
    ### PHASE 1: DATASET CREATION ###
    logger.info("PHASE 1 -> DATASET CREATION")
    create_dataset(EXPERIMENT_FT_DIR, TRAIN_REPLICAS, CROP_SIZE, EXP_METADATA)
    mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, EXP_METADATA)
    
    logger.info(f"Dataset creation completed!\n")
    
    ### PHASE 2: MODEL FINE-TUNING ###
    if "MODEL_FINE_TUNING" in FT_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 2 (Model Fine-Tuning): it has already been completed!\n")
    else:
        logger.info(f"PHASE 2 -> MODEL FINE-TUNING")
        set_random_seed(RANDOM_SEED)
        
        model, last_cp = load_model(EXPERIMENT_FT_DIR, "train", EXP_METADATA, FT_METADATA, DEVICE)
        
        train_dl = get_dataloader(EXPERIMENT_FT_DIR, "train", model.get_input_size(), mean_, std_, EXP_METADATA, FT_METADATA, DEVICE)
        val_dl = get_dataloader(EXPERIMENT_FT_DIR, "val", model.get_input_size(), mean_, std_, EXP_METADATA, FT_METADATA, DEVICE)
        start_ft = datetime.now()
        train_model(EXPERIMENT_FT_DIR, model, last_cp, train_dl, val_dl, FT_METADATA_PATH, FT_METADATA, DEVICE)
        end_ft = datetime.now()

        add_timestamp_to_ft_metadata(EXPERIMENT_ID, FT_METADATA, "MODEL_FINE_TUNING", str(end_ft))
        add_timestamp_to_ft_metadata(EXPERIMENT_ID, FT_METADATA, "FINE_TUNING_DURATION", str(end_ft - start_ft))
        del model, last_cp, train_dl, val_dl
        cleanup_memory(DEVICE)
        
        logger.info("Model fine-tuning completed successfully!\n")

    ### PHASE 3: MODEL TESTING ###
    if "MODEL_TESTING" in FT_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 3 (Model Testing): it has already been completed!\n")
    else:
        logger.info(f"PHASE 3 -> MODEL TESTING")
        
        mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, EXP_METADATA)
        model, _ = load_model(EXPERIMENT_FT_DIR, "test", EXP_METADATA, FT_METADATA, DEVICE)
        test_dl = get_dataloader(EXPERIMENT_FT_DIR, "test", model.get_input_size(), mean_, std_, EXP_METADATA, FT_METADATA, DEVICE)
        test_model(EXPERIMENT_FT_DIR, model, test_dl, EXP_METADATA, FT_METADATA, DEVICE)
        
        add_timestamp_to_ft_metadata(EXPERIMENT_ID, FT_METADATA, "MODEL_TESTING", str(datetime.now()))
        del model, test_dl
        cleanup_memory(DEVICE)
        logger.info("Model testing completed successfully!\n")

    if not KEEP_CROPS: remove_subdirectories(EXPERIMENT_FT_DIR, EXP_METADATA)
    
    logger.info(f"*** Experiment: {EXPERIMENT_ID} -> END OF FINE TUNING-PROCESS ***\n")
