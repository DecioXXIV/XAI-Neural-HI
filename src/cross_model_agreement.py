import numpy as np

from cli.arg_parsers import get_cm_agreement_args, validate_cm_agreement_args
from src.utils.logger import Logger
from src.utils.metadata.metadata_utils import get_xai_metadata
from src.utils.exp_comparison.evaluators import CrossModelAgreementEvaluator

logger = Logger()

if __name__ == "__main__":
    args = get_cm_agreement_args()
    EXP_ID1, EXP_ID2 = args.exp_id1, args.exp_id2
    XAI_ALGORITHM = args.xai_algorithm
    XAI_ENTRY1, XAI_ENTRY2 = args.xai_entry1, args.xai_entry2
    N_BOOTSTRAP, RANDOM_SEED = args.n_bootstrap, args.random_seed
    
    if RANDOM_SEED is None: RANDOM_SEED = int(np.random.randint(0, 2**32 - 1))
    
    XAI_EXP1_METADATA, XAI_EXP2_METADATA = get_xai_metadata(EXP_ID1), get_xai_metadata(EXP_ID2)
    validate_cm_agreement_args(EXP_ID1, EXP_ID2, XAI_ALGORITHM, XAI_ENTRY1, XAI_ENTRY2, XAI_EXP1_METADATA, XAI_EXP2_METADATA, N_BOOTSTRAP)
    
    cma_evaluator = CrossModelAgreementEvaluator(EXP_ID1, EXP_ID2, XAI_ALGORITHM, XAI_ENTRY1, XAI_ENTRY2, N_BOOTSTRAP, RANDOM_SEED)
    cma_evaluator()