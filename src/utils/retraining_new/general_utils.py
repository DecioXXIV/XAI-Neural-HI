import os, torch, json, shutil
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

from src.utils.logger import Logger
from src.utils.data.dataset_utils import get_dataset_instances
from src.utils.data.dataloaders import TestDataLoader
from src.utils.models.model_loader import FineTunedToRetrainModelLoader
from src.utils.fine_tuning.general_utils import create_dataset
from src.utils.faithfulness.patch_indexer import PatchIndexer
from src.utils.retraining import CROP_SET_CONFIGS
from src.utils.retraining.ft1_train_crops_coordinates_retriever import FT1TrainCropsCoordinatesRetriever
from src.utils.retraining.ft1_classes_centroids_sigmas_computer import FT1ClassesCentroidsSigmasComputer
from src.utils.retraining.crop_inference_executor import CropInferenceExecutor
from src.utils.retraining.crop_xai_agg_scores_computer import CropXaiAggScoresComputer

from src.utils.retraining_new.crop_classification_confidence_computer import CropClassificationConfidenceComputer
from src.utils.retraining_new.crop_xai_evidence_computer import CropXaiEvidenceComputer
from src.utils.retraining_new.crop_r2_retriever import CropR2Retriever
from src.utils.retraining_new.crop_ink_fraction_computer import CropInkFractionComputer
from src.utils.retraining_new.xai_guided_crop_selector import XaiGuidedCropSelector

logger = Logger()

def retrieve_original_dataset(base_dir: str, dataset: str, classes: List[str], crop_size: int):
    train_subdir = os.path.join(base_dir, "train")
    val_subdir = os.path.join(base_dir, "val")
    test_subdir = os.path.join(base_dir, "test")
    
    if os.path.exists(train_subdir) and os.path.exists(val_subdir) and os.path.exists(test_subdir):
        logger.warning(f"Skipping original dataset retrieval since '{train_subdir}', '{val_subdir}' and '{test_subdir}' already exist.")
        
    else: create_dataset(base_dir, dataset, classes, 1, crop_size)

def compute_ft2_crop_counts(base_dir: str, classes: List[str], original_ts_ratio: float, new_ts_ratio: float) -> Tuple[Dict[str, int], Dict[str, int]]:
    memory_crops, new_crops = {}, {}
    
    train_subdir = os.path.join(base_dir, "train")
    for cls in classes:
        class_files = os.listdir(os.path.join(train_subdir, cls))
        memory_crops[cls] = int(np.ceil(len(class_files) * original_ts_ratio))
        new_crops[cls] = int(np.ceil(len(class_files) * new_ts_ratio))
    
    return memory_crops, new_crops

def get_cls_to_crop(base_dir: str, ft_dir: str, crop_set: str) -> Tuple[Dict[str, str], Dict[str, int]]:
    with open(os.path.join(ft_dir, "class_to_idx.json"), 'r') as f:
        idx_to_cls = json.load(f)
    
    cls_to_crop_path = os.path.join(base_dir, f"cls_to_{crop_set}_crop.json")
    if os.path.exists(cls_to_crop_path):
        with open(cls_to_crop_path, 'r') as f: cls_to_crop = json.load(f)
    
    else:
        cls_to_crop = {}
        for cls in idx_to_cls.keys():
            if crop_set == "ft1_train": cls_train_crops = os.listdir(os.path.join(base_dir, "train", cls))
            elif crop_set == "xai_guided": cls_train_crops = os.listdir(os.path.join(base_dir, "xai_crops", cls))
            for crop in cls_train_crops:
                if crop_set == "ft1_train": cls_to_crop[os.path.join(base_dir, "train", cls, crop)] = cls
                elif crop_set == "xai_guided": cls_to_crop[os.path.join(base_dir, "xai_crops", cls, crop)] = cls
        
        with open(cls_to_crop_path, 'w') as f: json.dump(cls_to_crop, f, indent=4)
    
    return cls_to_crop, idx_to_cls

def retrieve_ft1_train_crops_coordinates(base_dir: str, dataset: str, classes: List[str], crop_size: int):
    coords_path = os.path.join(base_dir, "coords_to_ft1_train_crop.json")
    
    if os.path.exists(coords_path): logger.warning(f"Skipping coordinates retrieval for FT1 train crops: it has been already done!")
    else:
        logger.info("Retrieving coordinates for FT1 train crops...")
        coordinates_to_ft1_train_crop = FT1TrainCropsCoordinatesRetriever(base_dir, dataset, classes, crop_size)()
        with open(coords_path, 'w') as f: json.dump(coordinates_to_ft1_train_crop, f, indent=4)

def compute_memory_scores(base_dir: str, experiment_xai_dir: str, model: nn.Module, classes: List[str], batch_size: int, crop_size: int, mean_: List[float], std_: List[float], device: str):
    mem_scores_path = os.path.join(base_dir, "memory_scores.csv")
    
    if os.path.exists(mem_scores_path): logger.warning(f"Skipping memory scores computation: it has been already done!")
    
    else:
        logger.info("Computing memory scores for FT1 train crops...")
        
        mem_scores = pd.DataFrame()
        dataset, loader = TestDataLoader(os.path.join(base_dir, "train"), classes, batch_size, crop_size, mean_, std_, device).load_data()
        paths = [s[0] for s in dataset.samples]
        
        offset = 0
        cc, confidences = CropClassificationConfidenceComputer(model, device), []
        ec, green_evidences = CropXaiEvidenceComputer(base_dir, experiment_xai_dir, "ft1", "green"), []
        r2c, r2s = CropR2Retriever(experiment_xai_dir), []
        ifc, green_ink_fractions = CropInkFractionComputer(base_dir, experiment_xai_dir, "ft1", "green"), []
        for images, labels in tqdm(loader, desc="Batch Processing", position=0, leave=True, dynamic_ncols=True):
            n = images.size(0)
            batch_paths = paths[offset:offset+n]
            confidences.extend(cc.compute_confidence(images, labels))
            green_evidences.extend(ec(batch_paths))
            r2s.extend(r2c(batch_paths))
            green_ink_fractions.extend(ifc(batch_paths))
            
            offset += n
        
        mem_scores["crop"] = paths
        mem_scores["confidence"] = _adjust_confidences(confidences)
        mem_scores["green_evidence"] = green_evidences
        mem_scores["r2"] = r2s
        mem_scores["green_ink_fraction"] = green_ink_fractions
        mem_scores["memory"] = mem_scores["confidence"] * mem_scores["green_evidence"] * mem_scores["r2"] * mem_scores["green_ink_fraction"]
        mem_scores.to_csv(mem_scores_path, index=False, header=True)

def _adjust_confidences(confidences: List[float]) -> List[float]:
    positive_confidences = sorted([c for c in confidences if c > 0.0], reverse=True)
    if len(positive_confidences) >= 2:
        min_pos        = positive_confidences[-1]
        second_min_pos = positive_confidences[-2]
        eps            = second_min_pos - min_pos
        return [c if c > 0.0 else eps for c in confidences]
    return confidences
  
def extract_memory_crops(base_dir: str, dst_dir: str, classes: List[str], cls_to_crop: Dict[str, str], n_memory_crops_to_cls: Dict[str, int]) -> Dict[str, int]:
    with open(os.path.join(base_dir, "memory_scores.csv"), 'r') as f: mem_scores_df = pd.read_csv(f)
    crop_to_score = mem_scores_df.set_index("crop")["memory"].to_dict()
    
    cls_to_crops_list: Dict[str, List[str]] = defaultdict(list)
    for crop, cls in cls_to_crop.items(): cls_to_crops_list[cls].append(crop)
    
    for cls in classes:
        dst_class_dir = os.path.join(dst_dir, "train", cls)
        os.makedirs(dst_class_dir, exist_ok=True)
        
        cls_crops = cls_to_crops_list[cls]
        mem_scores_to_cls_crops = {crop: crop_to_score[crop] for crop in cls_crops}
        sorted_mem_scores_to_cls_crops = {k: v for k, v in sorted(mem_scores_to_cls_crops.items(), key=lambda item: item[1], reverse=True)}
        
        n_to_keep = n_memory_crops_to_cls[cls]
        selected_crops = list(sorted_mem_scores_to_cls_crops.keys())[:n_to_keep]
        for crop in selected_crops: shutil.copyfile(crop, os.path.join(dst_class_dir, os.path.basename(crop)))

def retrieve_xai_guided_crops(base_dir: str, experiment_xai_dir: str, dataset: str, classes: List[str], crop_size: int):
    coords_to_xai_crop_path = os.path.join(base_dir, "coords_to_xai_crop.json")
    
    if os.path.exists(coords_to_xai_crop_path):
        logger.warning(f"Skipping XAI-guided crops retrieval: it has been already done!")
    
    else:
        
        def _retrieve_xai_guided_crops_for_instance(inst_path: str) -> Tuple[List[str], List[Tuple[int, int, int, int]]]:
            cls          = os.path.basename(os.path.dirname(inst_path))
            page_name    = os.path.splitext(os.path.basename(inst_path))[0]
            page_xai_dir = os.path.join(experiment_xai_dir, page_name)

            with open(os.path.join(page_xai_dir, "aggregated_scores.json"), "r") as f: scores = json.load(f)

            segments     = np.load(os.path.join(page_xai_dir, "segments.npy"))
            bboxes       = PatchIndexer()(segments)
            padded_img   = Image.open(os.path.join(page_xai_dir, f"{page_name}_forexp.png")).convert("RGB")
            img_w, img_h = padded_img.size
            
            out_dir = os.path.join(base_dir, "xai_crops", cls)
            
            page_crop_paths, page_crop_coords = [], []
            half = crop_size // 2
            for patch_idx_str, score in scores.items():
                if score >= 0.0: continue

                coords = bboxes.get(str(patch_idx_str))
                if coords is None: continue
                p_left, p_top, p_right, p_bottom = coords["left"], coords["top"], coords["right"], coords["bottom"]
                cx, cy = (p_left + p_right) // 2, (p_top + p_bottom) // 2
                left, top = cx - half, cy - half
                right, bottom = left + crop_size - 1, top + crop_size - 1
                
                if left < 0 or top < 0 or right >= img_w or bottom >= img_h: continue

                out_path = os.path.join(out_dir, f"{page_name}_patch{patch_idx_str}.png")
                padded_img.crop((left, top, right + 1, bottom + 1)).save(out_path)

                page_crop_paths.append(out_path)
                page_crop_coords.append((left, top, right, bottom))

            return page_crop_paths, page_crop_coords
        
        logger.info("Retrieving XAI-guided crops...")
        for cls in classes: os.makedirs(os.path.join(base_dir, "xai_crops", cls), exist_ok=True)
        
        instance_paths = get_dataset_instances(dataset, classes, "train")
        coords_to_xai_crop = {}
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_retrieve_xai_guided_crops_for_instance, p): p for p in instance_paths}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Retrieving XAI-guided crops", dynamic_ncols=True):
                page_crop_paths, page_crop_coords = future.result()
                for path, coords in zip(page_crop_paths, page_crop_coords):
                    coords_to_xai_crop[path] = coords
        
        coords_to_xai_crop_path = os.path.join(base_dir, "coords_to_xai_crop.json")
        with open(coords_to_xai_crop_path, 'w') as f: json.dump(coords_to_xai_crop, f, indent=4)

def compute_openness_scores(base_dir: str, experiment_xai_dir: str, model: nn.Module, classes: List[str], batch_size: int, crop_size: int, mean_: List[float], std_: List[float], device: str):
    opennes_scores_path = os.path.join(base_dir, "openness_scores.csv")
    
    if os.path.exists(opennes_scores_path): logger.warning(f"Skipping openness scores computation: it has been already done!")
    
    else:
        logger.info("Computing openness scores for XAI-guided crops...")
        
        openness_scores = pd.DataFrame()
        dataset, loader = TestDataLoader(os.path.join(base_dir, "xai_crops"), classes, batch_size, crop_size, mean_, std_, device).load_data()
        paths = [s[0] for s in dataset.samples]
        
        offset = 0
        cc, difficulties = CropClassificationConfidenceComputer(model, device), []
        ec, red_evidences = CropXaiEvidenceComputer(base_dir, experiment_xai_dir, "xai_guided", "red"), []
        rifc, red_ink_fractions = CropInkFractionComputer(base_dir, experiment_xai_dir, "xai_guided", "red"), []
        for images, labels in tqdm(loader, desc="Batch Processing", position=0, leave=True, dynamic_ncols=True):
            n = images.size(0)
            batch_paths = paths[offset:offset + n]
            difficulties.extend(cc.compute_difficulty(images, labels))
            red_evidences.extend(ec(batch_paths))
            red_ink_fractions.extend(rifc(batch_paths))
            offset += n
        
        openness_scores["crop"] = paths
        openness_scores["difficulty"] = difficulties
        openness_scores["red_evidence"] = red_evidences
        openness_scores["red_ink_fraction"] = red_ink_fractions
        openness_scores["openness"] = openness_scores["difficulty"] * openness_scores["red_evidence"] * openness_scores["red_ink_fraction"]
        openness_scores.to_csv(opennes_scores_path, index=False, header=True)
        
def extract_xai_guided_crops(base_dir: str, dst_dir: str, experiment_xai_dir: str, classes: List[str], cls_to_crop: Dict[str, str], new_to_class: Dict[str, int]):
    crop_to_openness = pd.read_csv(os.path.join(base_dir, "openness_scores.csv")).set_index("crop")["openness"].to_dict()

    cls_to_crops_list: Dict[str, List[str]] = defaultdict(list)
    for crop, cls in cls_to_crop.items(): cls_to_crops_list[cls].append(crop)

    selector = XaiGuidedCropSelector(experiment_xai_dir, crop_to_openness)
    for cls in classes:
        dst_class_dir = os.path.join(dst_dir, "train", cls)
        os.makedirs(dst_class_dir, exist_ok=True)
        selected_crops = selector(cls_to_crops_list[cls], new_to_class[cls])
        for crop in selected_crops: shutil.copyfile(crop, os.path.join(dst_class_dir, os.path.basename(crop)))
        logger.info(f"Class '{cls}': {len(selected_crops)} XAI-guided crops retrieved.")
   
def extract_random_crops(dst_dir: str, experiment_xai_dir: str, dataset: str, classes: List[str], crop_size: int, n_to_class_new: Dict[str, int], random_seed: int):
    instance_paths = get_dataset_instances(dataset, classes, "train")
    rng = np.random.default_rng(random_seed)
    for cls in classes:
        dst_train_class_subdir = os.path.join(dst_dir, "train", cls)
        os.makedirs(dst_train_class_subdir, exist_ok=True)
        
        cls_instance_paths = [p for p in instance_paths if os.path.basename(os.path.dirname(p)) == cls]
        n_to_extract = n_to_class_new[cls]

        sampled_paths = rng.choice(cls_instance_paths, size=n_to_extract, replace=True)

        # Group sample indices by unique image path to open each image only once
        by_image: Dict[str, List[int]] = {}
        for idx, path in enumerate(sampled_paths): by_image.setdefault(path, []).append(idx)

        for inst_path, indices in by_image.items():
            page_name = os.path.splitext(os.path.basename(inst_path))[0]
            img = Image.open(os.path.join(experiment_xai_dir, page_name, f"{page_name}_forexp.png")).convert("RGB")
            img_w, img_h = img.size

            lefts = rng.integers(0, img_w - crop_size, size=len(indices))
            tops  = rng.integers(0, img_h - crop_size, size=len(indices))

            for i, idx in enumerate(indices):
                left, top = int(lefts[i]), int(tops[i])
                crop = img.crop((left, top, left + crop_size, top + crop_size))
                crop.save(os.path.join(dst_train_class_subdir, f"{page_name}_random{idx}.png"))

def load_ft_model(experiment_ft_dir: str, experiment_retrain_dir: str, model_name: str, classes: List[str], ft_mode: str, ch_layers: str, device: str, ft_metadata: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model_loader = FineTunedToRetrainModelLoader(experiment_ft_dir, experiment_retrain_dir, model_name, classes, ft_mode, ft_metadata)
    return model_loader(ch_layers.split(','), device)