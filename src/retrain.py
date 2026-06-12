import os, torch, shutil
from datetime import datetime

from cli.arg_parsers import get_retraining_args
from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_ft_metadata, get_experiment_metadata, get_retrain_metadata, initialize_retrain_metadata, add_timestamp_to_retrain_metadata
from src.utils.models.model_utils import load_model, train_model, test_model, setup_device, set_random_seed
from src.utils.fine_tuning.general_utils import get_train_rgb_mean_std, get_dataloader, remove_subdirectories
from src.utils.retrain.general_utils import retrieve_original_dataset, compute_ft2_crop_counts, retrieve_ft1_train_crops_coordinates, compute_memory_scores, extract_memory_crops, retrieve_xai_guided_crops, extract_random_crops, compute_openness_scores, extract_xai_guided_crops, load_ft_model

logger = Logger()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY, ORIGINAL_TS_RATIO, NEW_TS_RATIO, SELECTION_RULE, START_POINT, BATCH_SIZE, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, RANDOM_SEED, EPOCHS, TRAIN_TRANSFORMS, FT_MODE, KEEP_CROPS = get_retraining_args()
    
    EXP_METADATA, FT_METADATA = get_experiment_metadata(EXPERIMENT_ID), get_ft_metadata(EXPERIMENT_ID)
    RETRAIN_METADATA = get_retrain_metadata(EXPERIMENT_ID, FT_MODE, START_POINT, ORIGINAL_TS_RATIO, NEW_TS_RATIO, SELECTION_RULE, RANDOM_SEED)
    
    RETRAIN_METADATA, RETRAIN_METADATA_PATH = initialize_retrain_metadata(EXPERIMENT_ID, RETRAIN_METADATA, FT_METADATA, XAI_ALGORITHM, XAI_ENTRY, ORIGINAL_TS_RATIO, NEW_TS_RATIO, SELECTION_RULE, BATCH_SIZE, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, RANDOM_SEED, EPOCHS, TRAIN_TRANSFORMS, FT_MODE, START_POINT)
    
    EXPERIMENT_FT_DIR       = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning")
    EXPERIMENT_XAI_DIR      = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "xai", XAI_ALGORITHM, XAI_ENTRY)
    EXPERIMENT_RETRAIN_ROOT = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "retraining", XAI_ALGORITHM, XAI_ENTRY)
    EXPERIMENT_RETRAIN_DIR  = os.path.join(EXPERIMENT_RETRAIN_ROOT, f"{FT_MODE}-{START_POINT}", SELECTION_RULE, f"original{ORIGINAL_TS_RATIO}-new{NEW_TS_RATIO}", f"random_seed{RANDOM_SEED}")
    os.makedirs(EXPERIMENT_RETRAIN_DIR, exist_ok=True)
    
    mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, EXP_METADATA)
    
    DEVICE = setup_device()
    
    model, _ = load_model(EXPERIMENT_FT_DIR, "test", EXP_METADATA, FT_METADATA, DEVICE)
    model.to(DEVICE)
    model.eval()
    
    logger.info(f"*** Experiment: {EXPERIMENT_ID} -> START OF RETRAINING-PROCESS ***")
    logger.info(f"XAI Algorithm: {XAI_ALGORITHM} | XAI Entry: {XAI_ENTRY} | Selection Rule: {SELECTION_RULE}")
    logger.info(f"Original Training Set Ratio: {ORIGINAL_TS_RATIO} | New Training Set Ratio: {NEW_TS_RATIO} | Random Seed: {RANDOM_SEED}\n")
    
    retrieve_original_dataset(EXPERIMENT_RETRAIN_ROOT, EXP_METADATA, FT_METADATA)
    
    print()
    logger.info("PHASE 1 -> FT2 TRAINING SET CREATION")

    if SELECTION_RULE == "saliency":
        n_memory_crops_to_cls, n_new_crops_to_cls = compute_ft2_crop_counts(EXPERIMENT_RETRAIN_ROOT, ORIGINAL_TS_RATIO, NEW_TS_RATIO, EXP_METADATA)
        # Dict[str, int]: {class_name: num_crops}

        retrieve_ft1_train_crops_coordinates(EXPERIMENT_RETRAIN_ROOT, EXP_METADATA, FT_METADATA)
        compute_memory_scores(EXPERIMENT_RETRAIN_ROOT, XAI_ALGORITHM, EXPERIMENT_XAI_DIR, model, mean_, std_, EXP_METADATA, FT_METADATA, DEVICE)
        extract_memory_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_RETRAIN_DIR, n_memory_crops_to_cls, EXP_METADATA)

        retrieve_xai_guided_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_XAI_DIR, mean_, EXP_METADATA, FT_METADATA)
        compute_openness_scores(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_XAI_DIR, model, mean_, std_, EXP_METADATA, FT_METADATA, DEVICE)
        extract_xai_guided_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_RETRAIN_DIR, EXPERIMENT_XAI_DIR, n_new_crops_to_cls, EXP_METADATA)

    else: # SELECTION_RULE == "random"
        extract_random_crops(EXPERIMENT_RETRAIN_DIR, EXPERIMENT_FT_DIR, EXPERIMENT_XAI_DIR, ORIGINAL_TS_RATIO, NEW_TS_RATIO, RANDOM_SEED, mean_, EXP_METADATA, FT_METADATA)
    
    if "MODEL_FINE_TUNING" in RETRAIN_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 3 (Model Re-Training): it has already been completed!\n")
    else:
        logger.info(f"PHASE 3 -> MODEL RE-TRAINING")
        set_random_seed(RANDOM_SEED)
        
        if START_POINT == "from_zero": 
            model, last_cp = load_model(EXPERIMENT_RETRAIN_DIR, "train", EXP_METADATA, RETRAIN_METADATA, DEVICE)
        else: # START_POINT == "from_ft1"
            model, last_cp = load_ft_model(EXPERIMENT_FT_DIR, EXPERIMENT_RETRAIN_DIR, EXP_METADATA, RETRAIN_METADATA, DEVICE)
        
        train_dl = get_dataloader(EXPERIMENT_RETRAIN_DIR, "train", model.get_input_size(), mean_, std_, EXP_METADATA, RETRAIN_METADATA, DEVICE)
        val_dl = get_dataloader(EXPERIMENT_RETRAIN_ROOT, "val", model.get_input_size(), mean_, std_, EXP_METADATA, RETRAIN_METADATA, DEVICE)
        train_model(EXPERIMENT_RETRAIN_DIR, model, last_cp, train_dl, val_dl, RETRAIN_METADATA_PATH, RETRAIN_METADATA, DEVICE)
        
        add_timestamp_to_retrain_metadata(RETRAIN_METADATA, RETRAIN_METADATA_PATH, "MODEL_FINE_TUNING", str(datetime.now()))
        torch.cuda.empty_cache()
        
        logger.info("Model re-training completed successfully!\n")
    
    if "MODEL_TESTING" in RETRAIN_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 4 (Model Testing): it has already been completed!\n")
    else:
        logger.info(f"PHASE 4 -> MODEL TESTING")
        
        mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, EXP_METADATA)
        model, _ = load_model(EXPERIMENT_RETRAIN_DIR, "test", EXP_METADATA, RETRAIN_METADATA, DEVICE)
        test_dl = get_dataloader(EXPERIMENT_RETRAIN_ROOT, "test", model.get_input_size(), mean_, std_, EXP_METADATA, RETRAIN_METADATA, DEVICE)
        
        shutil.copyfile(os.path.join(EXPERIMENT_RETRAIN_ROOT, "n_crops_per_instance.json"), os.path.join(EXPERIMENT_RETRAIN_DIR, "n_crops_per_instance.json"))
        test_model(EXPERIMENT_RETRAIN_DIR, model, test_dl, EXP_METADATA, RETRAIN_METADATA, DEVICE)
        
        logger.info("Model testing completed successfully!\n")
        
        add_timestamp_to_retrain_metadata(RETRAIN_METADATA, RETRAIN_METADATA_PATH, "MODEL_TESTING", str(datetime.now()))
        torch.cuda.empty_cache()
        
        logger.info(f"*** Experiment: {EXPERIMENT_ID} -> END OF RETRAINING-PROCESS ***\n")
        
        if not KEEP_CROPS: remove_subdirectories(EXPERIMENT_RETRAIN_DIR, EXP_METADATA)
        