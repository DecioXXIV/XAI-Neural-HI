import os, PIL
import numpy as np
from abc import ABC, abstractmethod

class SegmentsHandler(ABC):
    def __init__(self): pass
    
    def __call__(self, padded_img: PIL.Image.Image, page_xai_dir: str):
        segments = self._load_segments(page_xai_dir)
        if segments is None: segments = self.generate_segments(padded_img)
        return segments
    
    def _load_segments(self, page_xai_dir: str):
        segments_path = os.path.join(page_xai_dir, "segments.npy")
        if os.path.exists(segments_path): return np.load(segments_path, allow_pickle=True)
        else: return None
    
    @abstractmethod
    def generate_segments(self, padded_img: PIL.Image.Image) -> np.ndarray: pass