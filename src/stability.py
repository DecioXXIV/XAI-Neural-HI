from cli.arg_parsers import get_stability_args
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_xai_metadata
from src.utils.exp_comparison.stability_utils import get_xai_entries_from_root_entry, validate_xai_entries
from src.utils.exp_comparison.evaluators import StabilityEvaluator

logger = Logger()

if __name__ == "__main__":
    args = get_stability_args()
    EXPERIMENT_ID, XAI_ALGORITHM, XAI_ROOT_ENTRY = args.experiment_id, args.xai_algorithm, args.xai_root_entry
    
    XAI_METADATA = get_xai_metadata(EXPERIMENT_ID)
    
    xai_entries = get_xai_entries_from_root_entry(XAI_ROOT_ENTRY, XAI_ALGORITHM, XAI_METADATA)
    validate_xai_entries(xai_entries, XAI_ROOT_ENTRY)
    
    stability_evaluator = StabilityEvaluator(EXPERIMENT_ID, XAI_ALGORITHM, XAI_ROOT_ENTRY, xai_entries)
    stability_evaluator()
