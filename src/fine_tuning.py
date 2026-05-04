import os, torch, random
import numpy as np

from datetime import datetime

from cli.arg_parsers import get_ft_args
from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_experiment_metadata, get_ft_metadata, initialize_ft_metadata, add_timestamp_to_ft_metadata
from src.utils.fine_tuning.general_utils import create_dataset, get_train_rgb_mean_std, get_dataloader, remove_subdirectories
from src.utils.models.model_utils import load_model, train_model, test_model

logger = Logger()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    EXPERIMENT_ID, METRIC, CH_LAYERS, CROP_SIZE, BATCH_SIZE, OPTIMIZER, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, TRAIN_REPLICAS, RANDOM_SEED, EPOCHS, FT_MODE, KEEP_CROPS = get_ft_args()
    
    EXP_METADATA = get_experiment_metadata(EXPERIMENT_ID)
    FT_METADATA = get_ft_metadata(EXPERIMENT_ID)
    FT_METADATA = initialize_ft_metadata(EXPERIMENT_ID, FT_METADATA, METRIC, CH_LAYERS, CROP_SIZE, BATCH_SIZE, OPTIMIZER, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, TRAIN_REPLICAS, RANDOM_SEED, EPOCHS, FT_MODE)
    
    EXPERIMENT_FT_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning")
    os.makedirs(EXPERIMENT_FT_DIR, exist_ok=True)
    
    MODEL_NAME, DATASET, CLASSES = EXP_METADATA.get("MODEL_NAME"), EXP_METADATA.get("DATASET"), EXP_METADATA.get("CLASSES")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        n_devices = torch.cuda.device_count()
        logger.info(f"Device(s): {[torch.cuda.get_device_name(i) for i in range(n_devices)]}")
    
    logger.info(f"*** Experiment: {EXPERIMENT_ID} -> START OF FINE TUNING-PROCESS ***\n")
    
    ### PHASE 1: DATASET CREATION ###
    logger.info("PHASE 1 -> DATASET CREATION")
    create_dataset(EXPERIMENT_FT_DIR, DATASET, CLASSES, TRAIN_REPLICAS, CROP_SIZE)
    mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, DATASET, CLASSES)
    
    logger.info(f"Dataset creation completed!\n")
    
    ### PHASE 2: MODEL FINE-TUNING ###
    if "MODEL_FINE_TUNING" in FT_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 2 (Model Fine-Tuning): it has already been completed!\n")
    else:
        logger.info(f"PHASE 2 -> MODEL FINE-TUNING")
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
        
        model, last_cp = load_model(EXPERIMENT_ID, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "train", FT_METADATA)
        
        train_dl = get_dataloader(os.path.join(EXPERIMENT_FT_DIR, "train"), CLASSES, "train", BATCH_SIZE, model.get_input_size(), mean_, std_, DEVICE, RANDOM_SEED)
        val_dl = get_dataloader(os.path.join(EXPERIMENT_FT_DIR, "val"), CLASSES, "val", BATCH_SIZE, model.get_input_size(), mean_, std_, DEVICE)
        train_model(EXPERIMENT_ID, model, train_dl, val_dl, DEVICE, FT_METADATA, last_cp)

        add_timestamp_to_ft_metadata(EXPERIMENT_ID, FT_METADATA, "MODEL_FINE_TUNING", str(datetime.now()))
        torch.cuda.empty_cache()
        
        logger.info("Model fine-tuning completed successfully!\n")
    
    ### PHASE 3: MODEL TESTING ###
    if "MODEL_TESTING" in FT_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 3 (Model Testing): it has already been completed!\n")
    else:
        logger.info(f"PHASE 3 -> MODEL TESTING")
        
        mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, DATASET, CLASSES)
        model, _ = load_model(EXPERIMENT_ID, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "test", FT_METADATA)
        test_dl = get_dataloader(os.path.join(EXPERIMENT_FT_DIR, "test"), CLASSES, "test", BATCH_SIZE, model.get_input_size(), mean_, std_, DEVICE)
        
        test_model(EXPERIMENT_ID, model, test_dl, DEVICE, FT_METADATA, EXP_METADATA)
        
        logger.info("Model testing completed successfully!\n")
        
        logger.info(f"PHASE 4 -> DATA & METADATA HANDLING")
        if not KEEP_CROPS: remove_subdirectories(EXPERIMENT_FT_DIR, DATASET, CLASSES)
        
        add_timestamp_to_ft_metadata(EXPERIMENT_ID, FT_METADATA, "MODEL_TESTING", str(datetime.now()))
        torch.cuda.empty_cache()
        
        logger.info(f"*** Experiment: {EXPERIMENT_ID} -> END OF FINE TUNING-PROCESS ***\n")