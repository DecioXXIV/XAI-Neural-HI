import os, json
import numpy as np
import pandas as pd
import torch.nn as nn
import multiprocessing as mp
from typing import Dict, Any, List
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from src.utils.logger import Logger
from src.utils.explain.xai_image_preprocessor import XaiImagePreprocessor
from src.utils.explain.xai_visual_explanation_builder import XaiVisualExplanationBuilder
from src.explainers.base_explainers import BaseExplainer
from src.explainers.lime_explainer import LimeExplainer
from src.explainers.glime_binomial_explainer import GLimeBinomialExplainer
from src.explainers.occlusion_explainer import OcclusionExplainer

logger = Logger()

def setup_explainer(xai_algorithm: str, xai_entry: str, model: nn.Module, mean_: List[float], std_: List[float], ft_metadata: Dict[str, Any], xai_metadata: Dict[str, Any], device: str) -> Any:
    logger.info(f"Setting up the {xai_algorithm} Explainer...")
    
    if xai_algorithm == "Lime": return LimeExplainer(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)
    elif xai_algorithm == "GLimeBinomial": return GLimeBinomialExplainer(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)
    elif xai_algorithm == "Occlusion": return OcclusionExplainer(xai_entry, model, mean_, std_, ft_metadata, xai_metadata, device)

def execute_pages_preprocessing(instance_paths: List[str], crop_size: int, mean_: List[float], seg_type: str, patch_dim: int | None, experiment_xai_dir: str, xai_instances_metadata: Dict[str, Any]):
    img_preprocessor = XaiImagePreprocessor(crop_size, mean_)

    def process_instance(instance_path: str):
        page_name = os.path.basename(instance_path).split(".")[0]
        if page_name not in xai_instances_metadata["INSTANCES"]:
            page_xai_dir = os.path.join(experiment_xai_dir, page_name)
            os.makedirs(page_xai_dir, exist_ok=True)
            img = Image.open(instance_path).convert("RGB")
            img_preprocessor.execute_preprocessing(img, page_name, seg_type, patch_dim, page_xai_dir)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_instance, path): path for path in instance_paths}
        for _ in tqdm(as_completed(futures), total=len(instance_paths), desc="Preprocessing Pages", position=0, leave=True, dynamic_ncols=True): pass

def explain_instances(explainer: BaseExplainer, instances: List[str], labels: List[int], experiment_xai_dir: str, xai_instances_metadata: Dict[str, Any]):
    for instance, label in zip(instances, labels):
        instance_name = os.path.basename(instance).split(".")[0]
        page_xai_dir = os.path.join(experiment_xai_dir, instance_name)
        
        if instance_name not in xai_instances_metadata["INSTANCES"]:
            logger.info(f"Generating explanation for '{instance_name}' (Label: {label})...")
            page = Image.open(os.path.join(page_xai_dir, f"{instance_name}_forexp.png")).convert("RGB")
            crop_coordinates_df = pd.read_csv(os.path.join(page_xai_dir, "crop_coordinates.csv"), header=0)
            segments = np.load(os.path.join(page_xai_dir, "segments.npy"))
            explainer.explain_page(page, label, segments, crop_coordinates_df, page_xai_dir)
            
            xai_instances_metadata["INSTANCES"][instance_name] = str(datetime.now())
            with open(os.path.join(experiment_xai_dir, "xai_instances_metadata.json"), "w") as f: json.dump(xai_instances_metadata, f, indent=4)
        
        else: logger.warning(f"Skipping '{instance_name}': already explained!")

def _build_visualization_worker(args: tuple):
    instance, experiment_xai_dir, xai_instances_metadata = args
    instance_name = os.path.basename(instance).split(".")[0]
    page_xai_dir = os.path.join(experiment_xai_dir, instance_name)

    if instance_name in xai_instances_metadata["INSTANCES"] and not os.path.exists(os.path.join(page_xai_dir, f"{instance_name}_visual_exp.png")):
        segments = np.load(os.path.join(page_xai_dir, "segments.npy"))
        with open(os.path.join(page_xai_dir, "aggregated_scores.json"), "r") as f: scores = json.load(f)
        XaiVisualExplanationBuilder().build_visual_explanation(instance_name, page_xai_dir, segments, scores)

def build_exp_visualizations(instances: List[str], experiment_xai_dir: str, xai_instances_metadata: Dict[str, Any]):
    args = [(instance, experiment_xai_dir, xai_instances_metadata) for instance in instances]

    with ProcessPoolExecutor(mp_context=mp.get_context("spawn")) as executor:
        futures = {executor.submit(_build_visualization_worker, a): a[0] for a in args}
        for future in tqdm(as_completed(futures), total=len(instances), desc="Building Visualizations", position=0, leave=True, dynamic_ncols=True):
            future.result()