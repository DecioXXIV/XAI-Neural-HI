import sys
from typing import Dict, List, Any
from src.utils.logger import Logger

logger = Logger()

def get_xai_entries_from_root_entry(root_entry: str, xai_algorithm: str, xai_metadata: Dict[str, Any]) -> List[str]:
    subdict = xai_metadata.get(xai_algorithm, {})
    if subdict is not None: 
        all_xai_entries = list(subdict.keys())
        return [e for e in all_xai_entries if e.startswith(root_entry) and "END_TIMESTAMP" in subdict[e].keys()]
    else: return []

def validate_xai_entries(xai_entries: List[str], root_entry: str):
    error_trigger = False
    
    if xai_entries is None:
        logger.critical(f"root entry '{root_entry}' has no sub-entries")
        error_trigger = True
    
    if len(xai_entries) < 2:
        logger.critical(f"root entry '{root_entry}' has not enough sub-entries for a Stability assessment")
        error_trigger = True
    
    if error_trigger: sys.exit()