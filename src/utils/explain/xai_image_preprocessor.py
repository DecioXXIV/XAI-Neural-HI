import os, PIL
import numpy as np
import pandas as pd
import torchvision.transforms.v2 as v2
from typing import Dict, Any, List

from src.utils.fine_tuning.crop_retriever import CropRetriever
from src.utils.explain.square_patches_segments_handler import SquarePatchesSegmentsHandler
from src.utils.explain.ink_based_segments_handler import InkBasedSegmentsHandler

class XaiImagePreprocessor:
    def __init__(self, xai_algorithm: str, xai_entry: str, xai_metadata: Dict[str, Any], crop_size: int, mean_: List[float]):
        self.crop_size = crop_size
        self.mean_ = mean_
        self.crop_retriever = CropRetriever(crop_size)
        self.seg_params = xai_metadata[xai_algorithm][xai_entry]["HYPERPARAMETERS"]
    
    def execute_preprocessing(self, img: PIL.Image.Image, page_name: str, page_xai_dir: str):
        crop_coordinates_df = self.get_crop_coordinates_df(img, page_xai_dir)
        crop_coordinates_df.to_csv(os.path.join(page_xai_dir, "crop_coordinates.csv"), index=False, header=True)
        
        crops = self.get_crops_from_coordinates_df(img, crop_coordinates_df)
        for i in range(0, len(crops)):
            crop_name = crop_coordinates_df.iloc[i]["crop_id"]
            os.makedirs(os.path.join(page_xai_dir, "crops", crop_name), exist_ok=True)
            crops[i].save(os.path.join(page_xai_dir, "crops", crop_name, f"{crop_name}.png"))
        
        padded_img = self.produce_padded_page(img, crop_coordinates_df)
        padded_img.save(os.path.join(page_xai_dir, f"{page_name}_forexp.png"))
        
        seg_type = self.seg_params["seg_type"]
        segments = None
        if seg_type == "sq_patches": segments = SquarePatchesSegmentsHandler(self.seg_params["patch_dim"])(padded_img, page_xai_dir)
        elif seg_type == "ink_based": segments = InkBasedSegmentsHandler(self.seg_params["aggressiveness"], self.seg_params["granularity"], self.seg_params.get("grouping_method", ""))(padded_img, page_xai_dir)
        np.save(os.path.join(page_xai_dir, "segments.npy"), segments)
    
    def get_crop_coordinates_df(self, img: PIL.Image.Image, page_xai_dir: str) -> pd.DataFrame:
        crop_coordinates_df_path = os.path.join(page_xai_dir, "crop_coordinates.csv")
        if not os.path.exists(crop_coordinates_df_path): return self.crop_retriever.get_crops_df(img)
        else: return pd.read_csv(crop_coordinates_df_path, header=0)
    
    def get_crops_from_coordinates_df(self, img: PIL.Image.Image, crop_coordinates_df: pd.DataFrame) -> List[PIL.Image.Image]:
        return self.crop_retriever.get_crops_from_df(img, crop_coordinates_df)
    
    def produce_padded_page(self, img: PIL.Image.Image, crop_coordinates_df: pd.DataFrame) -> PIL.Image.Image:
        padded_w, padded_h = int(np.max(crop_coordinates_df["padding_right"])), int(np.max(crop_coordinates_df["padding_bottom"]))
        return v2.Pad((0, 0, padded_w, padded_h), padding_mode="edge")(img)