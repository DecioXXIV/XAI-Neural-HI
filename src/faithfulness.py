import os, json, torch
from datetime import datetime
from cli.arg_parsers import get_faithfulness_args, validate_faithfulness_args
from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_experiment_metadata, get_ft_metadata, get_xai_metadata, initialize_faithfulness_metadata, add_end_timestamp_to_faithfulness_metadata
from src.utils.models.model_utils import load_model
from src.utils.fine_tuning.general_utils import get_train_rgb_mean_std
from src.utils.faithfulness.explained_instances_retriever import ExplainedTestInstancesRetriever
from src.utils.faithfulness.general_utils import get_masker, compute_mask_rates, create_test_sets, remove_test_sets
from src.utils.faithfulness.faithfulness_evaluator import FaithfulnessEvaluator

logger = Logger()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = get_faithfulness_args()
    EXPERIMENT_ID = args.experiment_id
    XAI_ALGORITHM, XAI_ENTRY = args.xai_algorithm, args.xai_entry
    MASK_CEIL, MASK_STEP, MASK_RULE, PATCHES_COLOR = args.mask_ceil, args.mask_step, args.mask_rule, args.patches_color
    KEEP_MASKED_PAGES, KEEP_TEST_SETS = args.keep_masked_pages, args.keep_test_sets
    
    EXP_METADATA = get_experiment_metadata(EXPERIMENT_ID)
    FT_METADATA = get_ft_metadata(EXPERIMENT_ID)
    XAI_METADATA = get_xai_metadata(EXPERIMENT_ID)
    validate_faithfulness_args(EXPERIMENT_ID, XAI_ALGORITHM, MASK_CEIL, MASK_STEP, XAI_ENTRY, XAI_METADATA)
    
    FAITH_METADATA = initialize_faithfulness_metadata(EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY)
    FAITH_ENTRY = f"{MASK_RULE}-ceil{MASK_CEIL}-step{MASK_STEP}-{PATCHES_COLOR}"
    
    if FAITH_ENTRY not in FAITH_METADATA[XAI_ALGORITHM][XAI_ENTRY]:
        EXPERIMENT_FT_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning")
        EXPERIMENT_FAITH_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "faithfulness", XAI_ALGORITHM, XAI_ENTRY, FAITH_ENTRY)
        os.makedirs(EXPERIMENT_FAITH_DIR, exist_ok=True)
        
        # FIRST STEP: Instance Masking
        DATASET, CLASSES = EXP_METADATA.get("DATASET"), EXP_METADATA.get("CLASSES")
        FT_MODE, CH_LAYERS = FT_METADATA["HYPERPARAMETERS"]["ft_mode"], FT_METADATA["HYPERPARAMETERS"]["ch_layers"]
        SEG_TYPE = XAI_METADATA[XAI_ALGORITHM][XAI_ENTRY]["HYPERPARAMETERS"]["seg_type"]
        mean_, std_ = get_train_rgb_mean_std(os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning"), DATASET, CLASSES)
        
        XAI_INSTANCES_METADATA_PATH = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "xai", XAI_ALGORITHM, XAI_ENTRY, "xai_instances_metadata.json")
        with open(XAI_INSTANCES_METADATA_PATH, 'r') as f: XAI_INSTANCES_METADATA = json.load(f)
        
        instance_paths, instance_names = ExplainedTestInstancesRetriever(EXPERIMENT_ID, DATASET, CLASSES, XAI_ALGORITHM, XAI_ENTRY, XAI_INSTANCES_METADATA)()
        mask_rates = compute_mask_rates(MASK_CEIL, MASK_STEP)
        mean_, _ = get_train_rgb_mean_std(os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning"), DATASET, CLASSES)
        masking_color = torch.tensor(mean_).view(3, 1, 1)
        
        masker = get_masker(EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY, SEG_TYPE, MASK_RULE, mask_rates, PATCHES_COLOR, masking_color)
        masker(instance_paths, instance_names)
        
        # SECOND STEP: Faithfulness computation
        CROP_SIZE = FT_METADATA["HYPERPARAMETERS"]["crop_size"]
        create_test_sets(EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY, FAITH_ENTRY, mask_rates, XAI_INSTANCES_METADATA, DATASET, CLASSES, CROP_SIZE)
        
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
            n_devices = torch.cuda.device_count()
            logger.info(f"Device(s): {[torch.cuda.get_device_name(i) for i in range(n_devices)]}")

        MODEL_NAME = EXP_METADATA.get("MODEL_NAME")
        model, _ = load_model(EXPERIMENT_FT_DIR, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "test", FT_METADATA)
        model.to(DEVICE)
        model.eval()
        
        faith_evaluator = FaithfulnessEvaluator(EXPERIMENT_ID, EXPERIMENT_FT_DIR, XAI_ALGORITHM, XAI_ENTRY, FAITH_ENTRY, mask_rates, MASK_RULE)
        faith_evaluator(model, CLASSES, mean_, std_, EXP_METADATA, FT_METADATA, DEVICE)
        add_end_timestamp_to_faithfulness_metadata(EXPERIMENT_ID, FAITH_METADATA, XAI_ALGORITHM, XAI_ENTRY, FAITH_ENTRY, str(datetime.now()))
        
        logger.info(f"Faithfulness evaluation with configuration '{FAITH_ENTRY}' has been completed for '{EXPERIMENT_ID}' with '{XAI_ALGORITHM}' and '{XAI_ENTRY}'!\n")
        
        if not KEEP_TEST_SETS: remove_test_sets(EXPERIMENT_ID, XAI_ALGORITHM, XAI_ENTRY, FAITH_ENTRY)
    
    else:
        logger.warning(f"Faithfulness evaluation with configuration '{FAITH_ENTRY}' has already been performed for '{EXPERIMENT_ID}' with '{XAI_ALGORITHM}' and '{XAI_ENTRY}'!\n")