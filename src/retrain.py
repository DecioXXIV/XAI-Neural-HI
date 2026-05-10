import os, torch, random
import numpy as np
from datetime import datetime

from cli.arg_parsers import get_retraining_args
from src.utils.constants import METADATA_ROOT, EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_ft_metadata, get_experiment_metadata, get_retrain_metadata, initialize_retrain_metadata, add_timestamp_to_retrain_metadata
from src.utils.models.model_utils import load_model, train_model, test_model
from src.utils.fine_tuning.general_utils import get_train_rgb_mean_std, get_dataloader, remove_subdirectories
from src.utils.retraining.general_utils import retrieve_original_dataset, compute_old_new_crop_counts, get_cls_to_crop, retrieve_ft1_train_crops_coordinates, compute_xai_agg_scores, load_ft_model, compute_inference_probs, compute_crop_scores, extract_crops, extract_random_crops, retrieve_xai_guided_crops, extract_memory_crops, extract_xai_guided_crops

logger = Logger()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY, ORIGINAL_TS_RATIO, NEW_TS_RATIO, SELECTION_RULE, START_POINT, BATCH_SIZE, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, RANDOM_SEED, EPOCHS, TRAIN_TRANSFORMS, FT_MODE, KEEP_CROPS = get_retraining_args()
    
    EXP_METADATA, FT_METADATA = get_experiment_metadata(EXPERIMENT_ID), get_ft_metadata(EXPERIMENT_ID)
    RETRAIN_METADATA = get_retrain_metadata(EXPERIMENT_ID, ORIGINAL_TS_RATIO, NEW_TS_RATIO, SELECTION_RULE, RANDOM_SEED)
    
    RETRAIN_METADATA, RETRAIN_METADATA_PATH = initialize_retrain_metadata(EXPERIMENT_ID, RETRAIN_METADATA, FT_METADATA, XAI_ALGORITHM, XAI_ENTRY, ORIGINAL_TS_RATIO, NEW_TS_RATIO, SELECTION_RULE, BATCH_SIZE, LR, LR_SCHEDULER, LR_FINAL_DECAY_RATIO, WEIGHT_DECAY, LABEL_SMOOTHING, EARLY_STOPPING, EARLY_STOPPING_PATIENCE, RANDOM_SEED, EPOCHS, TRAIN_TRANSFORMS, FT_MODE, START_POINT)
    
    EXPERIMENT_FT_DIR       = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning")
    EXPERIMENT_XAI_DIR      = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "xai", XAI_ALGORITHM, XAI_ENTRY)
    EXPERIMENT_RETRAIN_ROOT = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "retraining", XAI_ALGORITHM, XAI_ENTRY)
    EXPERIMENT_RETRAIN_DIR  = os.path.join(EXPERIMENT_RETRAIN_ROOT, f"{FT_MODE}-{START_POINT}", SELECTION_RULE, f"original{ORIGINAL_TS_RATIO}-new{NEW_TS_RATIO}", f"random_seed{RANDOM_SEED}")
    os.makedirs(EXPERIMENT_RETRAIN_DIR, exist_ok=True)
    
    MODEL_NAME, DATASET, CLASSES = EXP_METADATA.get("MODEL_NAME"), EXP_METADATA.get("DATASET"), EXP_METADATA.get("CLASSES")
    CH_LAYERS, CROP_SIZE, TRAIN_REPLICAS = FT_METADATA["HYPERPARAMETERS"]["ch_layers"], FT_METADATA["HYPERPARAMETERS"]["crop_size"], FT_METADATA["HYPERPARAMETERS"]["train_replicas"]
    mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, DATASET, CLASSES)
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
        n_devices = torch.cuda.device_count()
        logger.info(f"Device(s): {[torch.cuda.get_device_name(i) for i in range(n_devices)]}")
    
    model, _ = load_model(EXPERIMENT_FT_DIR, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "test", FT_METADATA)
    model.to(DEVICE)
    model.eval()
    
    logger.info(f"*** Experiment: {EXPERIMENT_ID} -> START OF RETRAINING-PROCESS ***")
    logger.info(f"XAI Algorithm: {XAI_ALGORITHM} | XAI Entry: {XAI_ENTRY} | Selection Rule: {SELECTION_RULE}")
    logger.info(f"Original Training Set Ratio: {ORIGINAL_TS_RATIO} | New Training Set Ratio: {NEW_TS_RATIO} | Random Seed: {RANDOM_SEED}\n")
    
    retrieve_original_dataset(EXPERIMENT_RETRAIN_ROOT, DATASET, CLASSES, CROP_SIZE)
    
    # n_to_class_old, n_to_class_new = compute_old_new_crop_counts(EXPERIMENT_RETRAIN_ROOT, CLASSES, ORIGINAL_TS_RATIO, NEW_TS_RATIO)
    # n_to_class_xyz: Dict[str, int] = {class_name: num_crops}
    
    cls_to_ft1_train_crop, idx_to_cls = get_cls_to_crop(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_FT_DIR, "ft1_train")
    # cls_to_crop: Dict[str, str] = {crop_path: class_name}
    # idx_to_cls: Dict[str, int] = {class_name: class_idx}

    ### PHASE 1: MEMORY CROPS EXTRACTION ###
    print()
    logger.info("PHASE 1 -> MEMORY CROPS EXTRACTION")
    
    retrieve_ft1_train_crops_coordinates(EXPERIMENT_RETRAIN_ROOT, DATASET, CLASSES, CROP_SIZE)
    compute_xai_agg_scores(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_XAI_DIR, "memory")
    # compute_centroids_and_sigmas_for_ft1_classes(EXPERIMENT_RETRAIN_ROOT, model, CLASSES, mean_, std_, DEVICE, BATCH_SIZE, CROP_SIZE)
    compute_inference_probs(EXPERIMENT_RETRAIN_ROOT, model, CLASSES, BATCH_SIZE, CROP_SIZE, mean_, std_, DEVICE, "memory")
    
    compute_crop_scores(EXPERIMENT_RETRAIN_ROOT, cls_to_ft1_train_crop, "memory")
    
    # extract_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_RETRAIN_DIR, CLASSES, cls_to_ft1_train_crop, n_to_class_old, "memory")
    new_to_class = extract_memory_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_RETRAIN_DIR, CLASSES, cls_to_ft1_train_crop, ORIGINAL_TS_RATIO, NEW_TS_RATIO)
    
    ### PHASE 2: NEW CROPS EXTRACTION (XAI-GUIDED) ###
    print()
    logger.info("PHASE 2 -> NEW CROPS EXTRACTION (XAI-GUIDED)")
    
    if SELECTION_RULE == "saliency":
        retrieve_xai_guided_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_XAI_DIR, DATASET, CLASSES, CROP_SIZE)
        cls_to_xai_guided_crop, _ = get_cls_to_crop(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_FT_DIR, "xai_guided")
    
        compute_xai_agg_scores(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_XAI_DIR, "openness")
        compute_inference_probs(EXPERIMENT_RETRAIN_ROOT, model, CLASSES, BATCH_SIZE, CROP_SIZE, mean_, std_, DEVICE, "openness")
    
        compute_crop_scores(EXPERIMENT_RETRAIN_ROOT, cls_to_xai_guided_crop, "openness")
        
        # extract_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_RETRAIN_DIR, CLASSES, cls_to_xai_guided_crop, n_to_class_new, "openness")
        extract_xai_guided_crops(EXPERIMENT_RETRAIN_ROOT, EXPERIMENT_RETRAIN_DIR, CLASSES, cls_to_xai_guided_crop, new_to_class)
    
    else: # SELECTION_RULE == "random"
        extract_random_crops(EXPERIMENT_RETRAIN_DIR, EXPERIMENT_XAI_DIR, DATASET, CLASSES, CROP_SIZE, new_to_class, RANDOM_SEED)
    
    if "MODEL_FINE_TUNING" in RETRAIN_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 3 (Model Re-Training): it has already been completed!\n")
    else:
        logger.info(f"PHASE 3 -> MODEL RE-TRAINING")
        if RANDOM_SEED is not None:
            random.seed(RANDOM_SEED)
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)
        
        if START_POINT == "from_zero": model, last_cp = load_model(EXPERIMENT_RETRAIN_DIR, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "train", RETRAIN_METADATA)
        else: # START_POINT == "from_ft1"
            model, last_cp = load_ft_model(EXPERIMENT_FT_DIR, EXPERIMENT_RETRAIN_DIR, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, RETRAIN_METADATA)
        
        train_dl = get_dataloader(os.path.join(EXPERIMENT_RETRAIN_DIR, "train"), CLASSES, "train", BATCH_SIZE, model.get_input_size(), mean_, std_, DEVICE, RANDOM_SEED, TRAIN_TRANSFORMS)
        val_dl = get_dataloader(os.path.join(EXPERIMENT_RETRAIN_ROOT, "val"), CLASSES, "val", BATCH_SIZE, model.get_input_size(), mean_, std_, DEVICE)
        train_model(EXPERIMENT_RETRAIN_DIR, model, train_dl, val_dl, DEVICE, RETRAIN_METADATA, RETRAIN_METADATA_PATH, last_cp)
        
        add_timestamp_to_retrain_metadata(RETRAIN_METADATA, RETRAIN_METADATA_PATH, "MODEL_FINE_TUNING", str(datetime.now()))
        torch.cuda.empty_cache()
        
        logger.info("Model re-training completed successfully!\n")
    
    if "MODEL_TESTING" in RETRAIN_METADATA["TIMESTAMPS"]:
        logger.warning("Skipping PHASE 4 (Model Testing): it has already been completed!\n")
    else:
        logger.info(f"PHASE 4 -> MODEL TESTING")
        
        mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, DATASET, CLASSES)
        model, _ = load_model(EXPERIMENT_RETRAIN_DIR, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "test", RETRAIN_METADATA)
        test_dl = get_dataloader(os.path.join(EXPERIMENT_RETRAIN_ROOT, "test"), CLASSES, "test", BATCH_SIZE, model.get_input_size(), mean_, std_, DEVICE)
        
        os.system(f"cp {os.path.join(EXPERIMENT_RETRAIN_ROOT, 'n_crops_per_instance.json')} {os.path.join(EXPERIMENT_RETRAIN_DIR, 'n_crops_per_instance.json')}")
        test_model(EXPERIMENT_RETRAIN_DIR, model, test_dl, DEVICE, RETRAIN_METADATA, EXP_METADATA)
        
        logger.info("Model testing completed successfully!\n")
        
        add_timestamp_to_retrain_metadata(RETRAIN_METADATA, RETRAIN_METADATA_PATH, "MODEL_TESTING", str(datetime.now()))
        torch.cuda.empty_cache()
        
        logger.info(f"*** Experiment: {EXPERIMENT_ID} -> END OF RETRAINING-PROCESS ***\n")
        
        if not KEEP_CROPS: remove_subdirectories(EXPERIMENT_RETRAIN_DIR, DATASET, CLASSES)