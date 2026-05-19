import os
from typing import Dict, List, Tuple
from tqdm import tqdm
from PIL import Image

from src.utils.data.dataset_utils import get_dataset_instances
from src.utils.fine_tuning.crop_retriever import CropRetriever

class FT1TrainCropsCoordinatesRetriever:
    def __init__(self, base_dir: str, dataset: str, classes: List[str], crop_size: int):
        self.base_dir = base_dir
        self.dataset = dataset
        self.classes = classes
        self.crop_retriever = CropRetriever(crop_size)
    
    def __call__(self) -> Dict[str, Tuple[int, int, int, int]]:
        coordinates_to_ft1_train_crop: Dict[str, Tuple[int, int, int, int]] = {}
        
        train_pages = get_dataset_instances(self.dataset, self.classes, "train")
        for page_path in tqdm(train_pages, desc="Processing FT1 train pages", dynamic_ncols=True):
            page_name = os.path.splitext(os.path.basename(page_path))[0]
            cls = os.path.basename(os.path.dirname(page_path))

            crops_df = self.crop_retriever.get_crops_df(Image.open(page_path))
            for _, row in crops_df.iterrows():
                crop_path = os.path.join(self.base_dir, "train", cls, f"{page_name}_cp1_{row['crop_id']}.png")
                left, top, right, bottom, pad_right, pad_bottom = row[["left_pixel", "top_pixel", "right_pixel", "bottom_pixel", "padding_right", "padding_bottom"]]
                coordinates_to_ft1_train_crop[crop_path] = (left, top, right+pad_right, bottom+pad_bottom)
        
        return coordinates_to_ft1_train_crop       