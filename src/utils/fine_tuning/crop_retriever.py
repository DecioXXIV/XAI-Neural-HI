import numpy as np
import pandas as pd
import torchvision.transforms.v2 as v2
from typing import List, Dict
from PIL import Image

class CropRetriever:
    def __init__(self, crop_size: int):
        self.crop_size = crop_size
    
    def _axis_starts(self, dim: int) -> List[int]:
        if dim <= self.crop_size:
            return [0]
    
        num_crops = int(np.ceil(dim / self.crop_size))
        max_start = dim - self.crop_size
        
        return [(i * max_start // (num_crops - 1)) for i in range(int(num_crops))]
        
    def _compute_crop_grid(self, img: Image.Image) -> List[Dict[str, int]]:
        img_w, img_h = img.size
        
        starts_x, starts_y = self._axis_starts(img_w), self._axis_starts(img_h)
        grid = []
        
        for left in starts_x:
            for top in starts_y:
                right = left + self.crop_size - 1
                bottom = top + self.crop_size - 1
                
                pad_right = max(0, right - img_w + 1)
                pad_bottom = max(0, bottom - img_h + 1)
                
                right, bottom = min(right, img_w - 1), min(bottom, img_h - 1)
                
                grid.append({"left_pixel": left, "top_pixel": top, "right_pixel": right, "bottom_pixel": bottom, "pad_right": pad_right, "pad_bottom": pad_bottom})
        
        return grid
    
    def get_crops(self, img: Image.Image) -> List[Image.Image]:
        grid, crops = self._compute_crop_grid(img), []
        for cell in grid:
            left, top, right, bottom = cell["left_pixel"], cell["top_pixel"], cell["right_pixel"], cell["bottom_pixel"]
            crop = img.crop((left, top, right + 1, bottom + 1)) # +1 because PIL cropping is non-inclusive of the right and bottom edges
            if cell["pad_right"] > 0 or cell["pad_bottom"] > 0:
                crop = v2.Pad((0, 0, cell["pad_right"], cell["pad_bottom"]), padding_mode="edge")(crop)
            crops.append(crop)
        
        return crops

    def get_crops_df(self, img: Image.Image) -> pd.DataFrame:
        grid = self._compute_crop_grid(img)
        rows = []
        id_width = len(str(len(grid)))
        
        for crop_n, cell in enumerate(grid):
            crop_id = f"crop{crop_n+1:0{id_width}d}"
            rows.append({
                "crop_id": crop_id,
                "left_pixel": cell["left_pixel"],
                "top_pixel": cell["top_pixel"],
                "right_pixel": cell["right_pixel"],
                "bottom_pixel": cell["bottom_pixel"],
                "padding_right": cell["pad_right"],
                "padding_bottom": cell["pad_bottom"]
            })
        
        return pd.DataFrame(rows)
    
    def get_crops_from_df(self, img: Image.Image, crops_df: pd.DataFrame) -> List[Image.Image]:
        crops = []
        
        for _, row in crops_df.iterrows():
            left, top, right, bottom = row[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel"]]
            crop = img.crop((left, top, right + 1, bottom + 1)) # +1 because PIL cropping is non-inclusive of the right and bottom edges
            pad_right, pad_bottom = row[["padding_right", "padding_bottom"]]
            if pad_right > 0 or pad_bottom > 0:
                crop = v2.Pad((0, 0, pad_right, pad_bottom), padding_mode="edge")(crop)
            crops.append(crop)
        
        return crops