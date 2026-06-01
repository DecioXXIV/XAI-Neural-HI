import PIL
import numpy as np

from src.utils.explain.segments_handler import SegmentsHandler

class SquarePatchesSegmentsHandler(SegmentsHandler):
    def __init__(self, patch_dim: int):
        super().__init__()
        self.patch_dim = patch_dim
    
    def generate_segments(self, padded_img: PIL.Image.Image) -> np.ndarray:
        w, h = padded_img.size

        num_cols = (w // self.patch_dim) + 1
        num_rows = (h // self.patch_dim) + 1
        n_patches = num_cols * num_rows
        
        patch_idxs = np.arange(n_patches, dtype=np.int32)
        seg_array = patch_idxs.reshape((num_rows, num_cols)).repeat(self.patch_dim, axis=0).repeat(self.patch_dim, axis=1)
        
        seg_array = seg_array[:h, :w]
        
        # Rename of segments to be consecutive integers starting from 0
        unique_vals = np.unique(seg_array)
        return np.searchsorted(unique_vals, seg_array)