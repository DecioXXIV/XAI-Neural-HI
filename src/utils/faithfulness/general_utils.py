import os, torch, shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List
from PIL import Image
from tqdm import tqdm

from src.utils.constants import DATA_ROOT, EXPERIMENTS_ROOT
from src.utils.logger import Logger
from src.utils.fine_tuning.crop_retriever import CropRetriever
from src.maskers.image_masker import ImageMasker
from src.maskers.sq_patches_image_maskers import SqPatchesSaliencyImageMasker, SqPatchesRandomImageMasker
from src.maskers.ink_based_image_maskers import InkBasedSaliencyImageMasker, InkBasedRandomImageMasker

logger = Logger()

def compute_mask_rates(mask_ceil, mask_step) -> List[float]:
    mask_rates, current_mr = [], mask_step
    
    while current_mr <= mask_ceil:
        current_mr = round(current_mr, 5)
        mask_rates.append(current_mr)
        current_mr += mask_step
    
    return mask_rates

def get_masker(experiment_id: str, xai_algorithm: str, xai_entry: str, seg_type: str, mask_rule: str, mask_rates: List[float], patches_color: str, masking_color: torch.Tensor, global_seed: int | None) -> ImageMasker:
    maskers = {
        "sq_patches" : {
            "saliency": SqPatchesSaliencyImageMasker,
            "random":   SqPatchesRandomImageMasker,
        },
        "ink_based": {
            "saliency": InkBasedSaliencyImageMasker,
            "random":   InkBasedRandomImageMasker,
        },
    }
    
    if seg_type not in maskers: raise ValueError(f"Unknown seg_type '{seg_type}'. Expected one of: {list(maskers.keys())}")
    if mask_rule not in maskers[seg_type]: raise ValueError(f"Unknown mask_rule '{mask_rule}'. Expected one of: {list(maskers[seg_type].keys())}")
    
    return maskers[seg_type][mask_rule](experiment_id, xai_algorithm, xai_entry, mask_rates, patches_color, masking_color, global_seed)

def create_test_sets(experiment_id: str, xai_algorithm: str, xai_entry: str, faith_entry: str, mask_rates: List[float], xai_instances_metadata: Dict[str, Any], dataset: str, classes: List[str], crop_size: int):
    instance_names = list(xai_instances_metadata["INSTANCES"].keys())

    # Map each instance name to its class by scanning the test split of the dataset
    instance_to_class: Dict[str, str] = {}
    for cls in classes:
        cls_dir = os.path.join(DATA_ROOT, dataset, "test", cls)
        for fname in os.listdir(cls_dir):
            instance_to_class[os.path.splitext(fname)[0]] = cls

    all_mask_rates = [0.0] + mask_rates
    test_sets_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "faithfulness", xai_algorithm, xai_entry, faith_entry, "test_sets")

    for mr in all_mask_rates:
        for cls in classes:
            os.makedirs(os.path.join(test_sets_dir, str(mr), cls), exist_ok=True)

    crop_retriever = CropRetriever(crop_size)
    xai_entry_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "xai", xai_algorithm, xai_entry)
    masked_pages_root = os.path.join(EXPERIMENTS_ROOT, experiment_id, "masked_images", xai_algorithm, xai_entry, faith_entry)

    def process_instance(instance_name: str, mr: float):
        cls = instance_to_class.get(instance_name)
        if cls is None:
            logger.warning(f"Class not found for instance '{instance_name}'. Skipping.")
            return

        if mr == 0.0:
            img_path = os.path.join(xai_entry_dir, instance_name, f"{instance_name}_forexp.png")
        else:
            img_path = os.path.join(masked_pages_root, f"mask_rate{mr}", f"{instance_name}.png")

        img = Image.open(img_path).convert("RGB")
        crops = crop_retriever.get_crops(img)

        out_dir = os.path.join(test_sets_dir, str(mr), cls)
        id_pad_width = len(str(len(crops)))
        for n, crop in enumerate(crops):
            crop.save(os.path.join(out_dir, f"{instance_name}_crop{n+1:0{id_pad_width}d}.png"))

    tasks = [(name, mr) for mr in all_mask_rates for name in instance_names]
    logger.info(f"*** CREATING TEST SETS -> Experiment: {experiment_id} | XAI: {xai_algorithm}/{xai_entry} | Faithfulness: {faith_entry} ***")

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_instance, name, mr): (name, mr) for name, mr in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Creating test sets", position=0, leave=True, dynamic_ncols=True):
            future.result()

    logger.info(f"*** TEST SETS CREATED SUCCESSFULLY ***")

def remove_test_sets(experiment_id: str, xai_algorithm: str, xai_entry: str, faith_entry: str):
    test_sets_dir = os.path.join(EXPERIMENTS_ROOT, experiment_id, "faithfulness", xai_algorithm, xai_entry, faith_entry, "test_sets")
    shutil.rmtree(test_sets_dir, ignore_errors=True)
