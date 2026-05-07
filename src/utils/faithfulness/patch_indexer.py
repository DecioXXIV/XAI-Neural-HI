import numpy as np
from typing import Dict

class PatchIndexer:
    def __init__(self): pass
    
    def __call__(self, segments: np.ndarray) -> Dict[str, Dict[str, int]]:
        bboxes = {}
        
        flat = segments.ravel()
        order = np.argsort(flat, kind="stable")
        ys_sorted, xs_sorted = np.unravel_index(order, segments.shape)
        
        unique_ids, starts = np.unique(flat[order], return_index=True)
        ends = np.append(starts[1:], len(flat))
        
        for i, id in enumerate(unique_ids):
            y_coords = ys_sorted[starts[i]:ends[i]]
            x_coords = xs_sorted[starts[i]:ends[i]]
            
            top, left = np.min(y_coords), np.min(x_coords)
            bottom, right = np.max(y_coords), np.max(x_coords)
            area = (bottom - top + 1) * (right - left + 1)
            
            bboxes[str(id)] = {"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "area": int(area)}
        
        return bboxes