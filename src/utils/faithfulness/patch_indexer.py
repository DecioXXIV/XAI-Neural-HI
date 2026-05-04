import numpy as np
from typing import Dict

class PatchIndexer:
    def __init__(self): pass
    
    def __call__(self, segments: np.ndarray) -> Dict[str, Dict[str, int]]:
        bboxes = {}
        
        for patch_id in np.unique(segments):
            positions = np.where(segments == patch_id)
            top, left = np.min(positions[0]), np.min(positions[1])
            bottom, right = np.max(positions[0]), np.max(positions[1])
            area = (bottom - top + 1) * (right - left + 1)
            
            bboxes[str(patch_id)] = {"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "area": int(area)}
        
        return bboxes