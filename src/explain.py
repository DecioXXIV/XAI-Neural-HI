import os, torch
from datetime import datetime

from cli.arg_parsers import get_explain_args
from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_experiment_metadata, get_ft_metadata, get_xai_instances_metadata, get_xai_metadata, initialize_xai_metadata, add_end_timestamp_to_xai_metadata
from src.utils.fine_tuning.general_utils import get_train_rgb_mean_std
from src.utils.models.model_utils import load_model, setup_device
from src.utils.explain.general_utils import setup_explainer, execute_pages_preprocessing, explain_instances, build_exp_visualizations
from src.utils.explain.instance_to_explain_retriever import InstanceToExplainRetriever

logger = Logger()
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    EXPERIMENT_ID, XAI_ALGORITHM, XAI_DETAILS, SUBSAMPLE, SAVE_SAMPLES, SEG_TYPE, PATCH_DIM, AGGRESSIVENESS, GRANULARITY, GROUPING_METHOD, NUM_SAMPLES, KERNEL_WIDTH = get_explain_args()
    
    EXP_METADATA = get_experiment_metadata(EXPERIMENT_ID)
    FT_METADATA = get_ft_metadata(EXPERIMENT_ID)
    XAI_METADATA = get_xai_metadata(EXPERIMENT_ID)
    XAI_METADATA, XAI_ENTRY = initialize_xai_metadata(EXPERIMENT_ID, XAI_METADATA, XAI_ALGORITHM, XAI_DETAILS, SEG_TYPE, PATCH_DIM, AGGRESSIVENESS, GRANULARITY, GROUPING_METHOD, NUM_SAMPLES, KERNEL_WIDTH)
    
    EXPERIMENT_FT_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "fine_tuning")
    DATASET, MODEL_NAME, CLASSES = EXP_METADATA.get("DATASET"), EXP_METADATA.get("MODEL_NAME"), EXP_METADATA.get("CLASSES")
    
    BATCH_SIZE = FT_METADATA["HYPERPARAMETERS"]["batch_size"]
    CROP_SIZE = FT_METADATA["HYPERPARAMETERS"]["crop_size"]
    CH_LAYERS = FT_METADATA["HYPERPARAMETERS"]["ch_layers"]
    FT_MODE = FT_METADATA["HYPERPARAMETERS"]["ft_mode"]
    mean_, std_ = get_train_rgb_mean_std(EXPERIMENT_FT_DIR, DATASET, CLASSES)
    
    EXPERIMENT_XAI_DIR = os.path.join(EXPERIMENTS_ROOT, EXPERIMENT_ID, "xai", XAI_ALGORITHM, XAI_ENTRY)
    os.makedirs(EXPERIMENT_XAI_DIR, exist_ok=True)
    
    DEVICE = setup_device()

    model, _ = load_model(EXPERIMENT_FT_DIR, MODEL_NAME, CLASSES, FT_MODE, CH_LAYERS, "test", DEVICE, FT_METADATA)
    model.to(DEVICE)
    model.eval()
    
    XAI_INSTANCES_METADATA = get_xai_instances_metadata(EXPERIMENT_XAI_DIR)
    explainer = setup_explainer(XAI_ALGORITHM, XAI_ENTRY, model, mean_, std_, FT_METADATA, XAI_METADATA, DEVICE)
    instance_paths, labels = InstanceToExplainRetriever(EXPERIMENT_ID, MODEL_NAME, DATASET, CLASSES, SUBSAMPLE)()
    execute_pages_preprocessing(instance_paths, CROP_SIZE, mean_, XAI_ALGORITHM, XAI_ENTRY, XAI_METADATA, EXPERIMENT_XAI_DIR, XAI_INSTANCES_METADATA)
    
    instances = [os.path.basename(path) for path in instance_paths]
    
    # Generate explanations for the retrieved instances
    explain_instances(explainer, instances, labels, EXPERIMENT_XAI_DIR, XAI_INSTANCES_METADATA)
    build_exp_visualizations(instances, EXPERIMENT_XAI_DIR, XAI_INSTANCES_METADATA)
    
    add_end_timestamp_to_xai_metadata(EXPERIMENT_ID, XAI_METADATA, XAI_ALGORITHM, XAI_ENTRY, str(datetime.now()))
    
    logger.info(f"All the requested Instances have been explained!")
    logger.info(f"*** Experiment: {EXPERIMENT_ID} -> END OF EXPLAINABILITY PROCESS ***\n")