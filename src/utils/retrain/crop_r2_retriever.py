import os, json
from typing import List

class CropR2Retriever:
    def __init__(self, experiment_xai_dir: str, xai_algorithm: str):
        self.experiment_xai_dir = experiment_xai_dir
        self.xai_algorithm = xai_algorithm
        self.page_data_cache = {}
    
    def __call__(self, paths: List[str]) -> List[float]:
        if self.xai_algorithm == "Occlusion":
            return [1.0] * len(paths)  # R² is always 1 for Occlusion (perfect fit)
        
        elif self.xai_algorithm in ("GLimeBinomial", "Lime"):
            r2s = []
        
            for path in paths:
                instance_name = os.path.basename(path)
                page_name, crop_name = instance_name.split("_")[0], instance_name.split("_")[-1].split(".")[0]
            
                if page_name in self.page_data_cache: r2s_dict = self.page_data_cache[page_name]
                else:
                    r2s_path = os.path.join(self.experiment_xai_dir, page_name, "crop_r2s.json")
                    with open(r2s_path, "r") as f: r2s_dict = json.load(f)
                    self.page_data_cache[page_name] = r2s_dict

                r2s.append(r2s_dict.get(crop_name, 0.0))
        
            return r2s