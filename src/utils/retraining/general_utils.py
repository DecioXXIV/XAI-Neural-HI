import os, torch, json, shutil
import torch.nn as nn
import numpy as np
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

logger = Logger()

def retrieve_original_dataset(base_dir: str, dataset: str, classes: List[str], crop_size: int):
    train_subdir = os.path.join(base_dir, "train")
    val_subdir = os.path.join(base_dir, "val")
    test_subdir = os.path.join(base_dir, "test")
    
    if os.path.exists(train_subdir) and os.path.exists(val_subdir) and os.path.exists(test_subdir):
        logger.warning(f"Skipping original dataset retrieval since '{train_subdir}', '{val_subdir}' and '{test_subdir}' already exist.")
        
    else: create_dataset(base_dir, dataset, classes, 1, crop_size)

def retrieve_ft1_train_crops_coordinates(base_dir: str, dataset: str, classes: List[str], crop_size: int):
    coords_path = os.path.join(base_dir, "coords_to_ft1_train_crop.json")
    
    if os.path.exists(coords_path): logger.warning(f"Skipping coordinates retrieval for FT1 train crops: it has been already done!")
    else:
        logger.info("Retrieving coordinates for FT1 train crops...")
        coordinates_to_ft1_train_crop = FT1TrainCropsCoordinatesRetriever(base_dir, dataset, classes, crop_size)()
        with open(coords_path, 'w') as f: json.dump(coordinates_to_ft1_train_crop, f, indent=4)
    
def compute_old_new_crop_counts(base_dir: str, classes: List[str], original_ts_ratio: float, new_ts_ratio: float) -> Tuple[Dict[str, int], Dict[str, int]]:
    n_to_class_old, n_to_class_new = {}, {}
    
    train_subdir = os.path.join(base_dir, "train")
    for cls in classes:
        class_files = os.listdir(os.path.join(train_subdir, cls))
        n_to_class_old[cls] = int(np.ceil(len(class_files) * original_ts_ratio))
        n_to_class_new[cls] = int(np.ceil(len(class_files) * new_ts_ratio))
    
    return n_to_class_old, n_to_class_new

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

def compute_xai_agg_scores(base_dir: str, experiment_xai_dir: str, score_type: str):
    cfg = CROP_SET_CONFIGS[score_type]
    output_path = os.path.join(base_dir, cfg["xai_agg_scores_file"])

    if os.path.exists(output_path):
        logger.warning(cfg["xai_agg_scores_skip_msg"])
    else:
        logger.info(cfg["xai_agg_scores_start_msg"])
        with open(os.path.join(base_dir, cfg["xai_agg_scores_coords_file"]), 'r') as f: coords = json.load(f)
        agg_scores = CropXaiAggScoresComputer(experiment_xai_dir, cfg["xai_agg_scores_score_sign"], cfg["xai_agg_scores_tqdm_desc"])(coords)
        with open(output_path, 'w') as f: json.dump(agg_scores, f, indent=4)

def compute_centroids_and_sigmas_for_ft1_classes(base_dir: str, model: nn.Module, classes: List[str], mean_: List[float], std_: List[float], device: str, batch_size: int, crop_size: int):
    ft1_centroids_path = os.path.join(base_dir, "ft1_centroids.pt")
    ft1_sigmas_path = os.path.join(base_dir, "ft1_sigmas.pt")
    
    if os.path.exists(ft1_centroids_path) and os.path.exists(ft1_sigmas_path):
        logger.warning(f"Skipping centroids and sigmas computation: it has been already done!")
    
    else:
        logger.info("Computing centroids and sigmas for FT1 train crops...")
        ft1_centroids, ft1_sigmas = FT1ClassesCentroidsSigmasComputer(base_dir, model, classes, mean_, std_, device, batch_size, crop_size)()
        torch.save(ft1_centroids, ft1_centroids_path)
        torch.save(ft1_sigmas, ft1_sigmas_path)

def compute_inference_probs(base_dir: str, model: nn.Module, classes: List[str], batch_size: int, crop_size: int, mean_: List[float], std_: List[float], device: str, score_type: str):
    cfg = CROP_SET_CONFIGS[score_type]
    probs_path = os.path.join(base_dir, cfg["probs_file"])
    if os.path.exists(probs_path):
        logger.warning(cfg["probs_skip_msg"])
    else:
        logger.info(cfg["probs_start_msg"])
        source_subdir = os.path.join(base_dir, cfg["probs_source_subdir"])
        dataset, loader = TestDataLoader(source_subdir, classes, batch_size, crop_size, mean_, std_, device).load_data()
        sample_paths = [s[0] for s in dataset.samples]
        sample_labels = [s[1] for s in dataset.samples]

        all_probs = CropInferenceExecutor(model, device).compute_probs(loader)

        probs_to_crop = {}
        for i in range(len(sample_paths)):
            p_true = float(all_probs[i, sample_labels[i]].item())
            probs_to_crop[sample_paths[i]] = (1.0 - p_true) if cfg["invert_prob"] else p_true

        with open(probs_path, 'w') as f: json.dump(probs_to_crop, f, indent=4)

def compute_crop_scores(base_dir: str, cls_to_crop: Dict[str, str], score_type: str):    
    cfg = CROP_SET_CONFIGS[score_type]
    output_path = os.path.join(base_dir, cfg["output_file"])
    
    if os.path.exists(output_path):
        logger.warning(cfg["skip_msg"])
    else:
        logger.info(cfg["start_msg"])
        scores = {crop: 1.0 for crop in cls_to_crop.keys()}
        
        with open(os.path.join(base_dir, cfg["probs_file"]), 'r') as f: probs = json.load(f)
        with open(os.path.join(base_dir, cfg["xai_agg_scores_file"]), 'r') as f: xai_agg_scores = json.load(f)
        
        for crop in tqdm(scores.keys(), desc=cfg["tqdm_desc"], position=0, leave=True, dynamic_ncols=True):
            scores[crop] = probs[crop] * xai_agg_scores[crop]
        
        with open(output_path, 'w') as f: json.dump(scores, f, indent=4)
    
def extract_memory_crops(base_dir: str, dst_dir: str, classes: List[str], cls_to_crop: Dict[str, str], original_ts_ratio: float) -> Dict[str, int]:
    with open(os.path.join(base_dir, "mem_to_t1_train_crop.json"), 'r') as f: memory_scores = json.load(f)
    
    n_to_keep = int(np.ceil(len(cls_to_crop) * original_ts_ratio))
    sorted_memory_scores = {k: v for k, v in sorted(memory_scores.items(), key=lambda item: item[1], reverse=True)}
    selected_crops = list(sorted_memory_scores.keys())[:n_to_keep]
    
    ft1_crops_to_cls = {c: len([crop for crop in cls_to_crop.keys() if cls_to_crop[crop] == c]) for c in classes}
    retrieved_to_cls = {c: 0 for c in classes}

    for crop in selected_crops:
        cls = cls_to_crop[crop]
        retrieved_to_cls[cls] += 1
        shutil.copyfile(crop, os.path.join(dst_dir, "train", cls, os.path.basename(crop)))
    
    for cls in classes: logger.info(f"Class '{cls}': {retrieved_to_cls[cls]} memory crops retrieved.")
    
    new_to_class = {c: ft1_crops_to_cls[c] - retrieved_to_cls[c] for c in classes}
    return new_to_class

def extract_xai_guided_crops(base_dir: str, dst_dir: str, classes: List[str], cls_to_crop: Dict[str, str], new_to_class: Dict[str, int]):
    with open(os.path.join(base_dir, "openness_to_xai_crop.json"), 'r') as f: openness_scores = json.load(f)
    
    for cls in classes:
        openness_scores_to_cls_crops = {crop: openness_scores[crop] for crop in cls_to_crop.keys() if cls_to_crop[crop] == cls}
        sorted_openness_scores_to_cls_crops = {k: v for k, v in sorted(openness_scores_to_cls_crops.items(), key=lambda item: item[1], reverse=True)}
        
        to_retrieve = list(sorted_openness_scores_to_cls_crops.keys())[:new_to_class[cls]]
        for crop in to_retrieve: shutil.copyfile(crop, os.path.join(dst_dir, "train", cls, os.path.basename(crop)))
        logger.info(f"Class '{cls}': {new_to_class[cls]} XAI-guided crops retrieved.")

def extract_crops(base_dir: str, dst_dir: str, classes: List[str], cls_to_crop: Dict[str, str], n_to_class: Dict[str, int], score_type: str):
    cfg = CROP_SET_CONFIGS[score_type]
    with open(os.path.join(base_dir, cfg["output_file"]), 'r') as f: scores = json.load(f)
    
    for c in tqdm(classes, desc=cfg["extract_tqdm_desc"], position=0, leave=True, dynamic_ncols=True):
        dst_train_class_subdir = os.path.join(dst_dir, "train", c)
        os.makedirs(dst_train_class_subdir, exist_ok=True)
        
        cls_crops = [crop for crop in cls_to_crop.keys() if cls_to_crop[crop] == c]
        scores_to_cls_crops = {crop: scores[crop] for crop in cls_crops}
        sorted_scores_to_cls_crops = {k: v for k, v in sorted(scores_to_cls_crops.items(), key=lambda item: item[1], reverse=True)}
        
        n_to_keep = n_to_class[c]
        to_retrieve = list(sorted_scores_to_cls_crops.keys())[:n_to_keep]
        for crop in to_retrieve: shutil.copyfile(crop, os.path.join(dst_train_class_subdir, os.path.basename(crop)))
    
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
                right, bottom = left + crop_size, top + crop_size
                
                if left < 0 or top < 0 or right > img_w or bottom > img_h: continue

                out_path = os.path.join(out_dir, f"{page_name}_patch{patch_idx_str}.png")
                padded_img.crop((left, top, right, bottom)).save(out_path)

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
    
def load_ft_model(experiment_ft_dir: str, experiment_retrain_dir: str, model_name: str, classes: List[str], ft_mode: str, ch_layers: str, ft_metadata: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    model_loader = FineTunedToRetrainModelLoader(experiment_ft_dir, experiment_retrain_dir, model_name, classes, ft_mode, ft_metadata)
    return model_loader(ch_layers.split(','))

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