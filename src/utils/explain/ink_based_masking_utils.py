import cv2
import numpy as np
from typing import Iterable

MASK_DILATION_RADIUS = 1
LOCAL_BACKGROUND_INITIAL_PADDING = 2
LOCAL_BACKGROUND_EXPANSION_STEP = 4
LOCAL_BACKGROUND_MAX_EXPANSIONS = 5
LOCAL_BACKGROUND_MIN_PIXELS = 16

def _validate_inputs(image_rgb: np.ndarray, segments: np.ndarray):
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3: raise ValueError("Ink replacement expects an RGB image with shape (H, W, 3).")
    if segments.ndim != 2: raise ValueError("Ink replacement expects a 2-D segment map.")
    if tuple(image_rgb.shape[:2]) != tuple(segments.shape): raise ValueError(f"Image shape {image_rgb.shape[:2]} does not match segments shape {segments.shape}.")

def _normalise_segment_ids(segment_ids: Iterable[int]) -> list[int]:
    ids = []
    for segment_id in segment_ids:
        segment_id = int(segment_id)
        if segment_id != 0 and segment_id not in ids: ids.append(segment_id)
    return ids

def _expanded_bbox(mask: np.ndarray, padding: int) -> tuple[int, int, int, int] | None:
    """
    Return an inclusive bounding box ordered as ``(top, bottom, left, right)``.
    """
    ys, xs = np.where(mask)
    if ys.size == 0: return None

    height, width = mask.shape
    padding = max(0, int(padding))
    top = max(0, int(ys.min()) - padding)
    bottom = min(height - 1, int(ys.max()) + padding)
    left = max(0, int(xs.min()) - padding)
    right = min(width - 1, int(xs.max()) + padding)
    return top, bottom, left, right

def _median_rgb(pixels: np.ndarray, dtype: np.dtype) -> np.ndarray:
    color = np.median(pixels, axis=0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(color), info.min, info.max).astype(dtype)
    return color.astype(dtype)

def _fallback_background_color(image_rgb: np.ndarray, segments: np.ndarray) -> np.ndarray:
    background_pixels = image_rgb[segments == 0]
    if background_pixels.size == 0:
        background_pixels = image_rgb.reshape(-1, image_rgb.shape[2])
    return _median_rgb(background_pixels, image_rgb.dtype)

def estimate_background_color_from_bbox(image_rgb: np.ndarray, segments: np.ndarray, left: int, top: int, right: int, bottom: int, initial_padding: int = LOCAL_BACKGROUND_INITIAL_PADDING, expansion_step: int = LOCAL_BACKGROUND_EXPANSION_STEP, max_expansions: int = LOCAL_BACKGROUND_MAX_EXPANSIONS, min_pixels: int = LOCAL_BACKGROUND_MIN_PIXELS) -> np.ndarray:
    """
    Estimate a local erase colour from a known, inclusive segment bounding box.
    """
    _validate_inputs(image_rgb, segments)

    height, width = segments.shape
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(width - 1, int(right)), min(height - 1, int(bottom))
    if left > right or top > bottom:
        return _fallback_background_color(image_rgb, segments)

    for expansion_idx in range(max_expansions + 1):
        padding = int(initial_padding) + expansion_idx * int(expansion_step)
        expanded_left = max(0, left - padding)
        expanded_top = max(0, top - padding)
        expanded_right = min(width, right + padding + 1)
        expanded_bottom = min(height, bottom + padding + 1)

        bbox_image = image_rgb[expanded_top:expanded_bottom, expanded_left:expanded_right]
        bbox_segments = segments[expanded_top:expanded_bottom, expanded_left:expanded_right]
        background_pixels = bbox_image[bbox_segments == 0]

        if len(background_pixels) >= min_pixels:
            return _median_rgb(background_pixels, image_rgb.dtype)

    return _fallback_background_color(image_rgb, segments)

def build_ink_replacement_mask(segments: np.ndarray, segment_ids: Iterable[int], dilation_radius: int = MASK_DILATION_RADIUS) -> np.ndarray:
    """
    Build the pixel mask to replace for selected ink segments.

    The selected ink pixels are always included. A small dilation is allowed to
    expand only into background pixels (label 0), which removes edge remnants
    without erasing neighbouring ink segments that were not selected.
    """
    target_ids = _normalise_segment_ids(segment_ids)
    if not target_ids: return np.zeros_like(segments, dtype=bool)

    target_mask = np.isin(segments, target_ids)
    if dilation_radius <= 0: return target_mask

    kernel_size = 2 * int(dilation_radius) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return target_mask | (dilated_mask & (segments == 0))

def build_ink_replacement_mask_in_bbox(segments: np.ndarray, segment_id: int, left: int, top: int, right: int, bottom: int, dilation_radius: int = MASK_DILATION_RADIUS) -> tuple[tuple[int, int, int, int], np.ndarray]:
    """
    Build a replacement mask only inside the selected segment ROI.

    Input and returned bounds are inclusive and ordered as
    ``(left, top, right, bottom)``.
    """
    if segments.ndim != 2:
        raise ValueError("Ink replacement expects a 2-D segment map.")

    height, width = segments.shape
    segment_id = int(segment_id)
    padding = max(0, int(dilation_radius))
    roi_left = max(0, int(left) - padding)
    roi_top = max(0, int(top) - padding)
    roi_right = min(width - 1, int(right) + padding)
    roi_bottom = min(height - 1, int(bottom) + padding)

    if segment_id == 0 or roi_left > roi_right or roi_top > roi_bottom:
        return (roi_left, roi_top, roi_right, roi_bottom), np.zeros((0, 0), dtype=bool)

    roi_segments = segments[roi_top:roi_bottom + 1, roi_left:roi_right + 1]
    target_mask = roi_segments == segment_id
    if padding <= 0 or not np.any(target_mask):
        return (roi_left, roi_top, roi_right, roi_bottom), target_mask

    kernel_size = 2 * padding + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    replacement_mask = target_mask | (dilated_mask & (roi_segments == 0))
    return (roi_left, roi_top, roi_right, roi_bottom), replacement_mask

def build_ink_masking_cache(image_rgb: np.ndarray, segments: np.ndarray, segment_ids: Iterable[int], dilation_radius: int = MASK_DILATION_RADIUS) -> dict[int, dict]:
    """
    Pre-compute local replacement data for each selected ink segment.
    """
    _validate_inputs(image_rgb, segments)
    masking_cache = {}

    for segment_id in _normalise_segment_ids(segment_ids):
        target_mask = segments == segment_id
        bbox = _expanded_bbox(target_mask, 0)
        if bbox is None: continue

        top, bottom, left, right = bbox
        roi, replacement_mask = build_ink_replacement_mask_in_bbox(segments, segment_id, left, top, right, bottom, dilation_radius=dilation_radius)
        replacement_color = estimate_background_color_from_bbox(image_rgb, segments, left, top, right, bottom)
        
        masking_cache[segment_id] = {
            "bbox": (left, top, right, bottom),
            "roi": roi,
            "replacement_mask": replacement_mask,
            "replacement_color": replacement_color,
        }

    return masking_cache

def replace_ink_segments_from_cache(target_rgb: np.ndarray, segment_ids: Iterable[int], masking_cache: dict[int, dict] | None) -> np.ndarray:
    """
    Replace selected ink segments using pre-computed crop-local data.
    """
    if target_rgb.ndim != 3 or target_rgb.shape[2] != 3:
        raise ValueError("Ink replacement expects an RGB image with shape (H, W, 3).")
    if masking_cache is None:
        raise ValueError("Ink replacement cache has not been initialized.")

    for segment_id in sorted(_normalise_segment_ids(segment_ids)):
        segment_data = masking_cache.get(segment_id)
        if segment_data is None: continue

        left, top, right, bottom = segment_data["roi"]
        replacement_mask = segment_data["replacement_mask"]
        target_roi = target_rgb[top:bottom + 1, left:right + 1]
        if tuple(target_roi.shape[:2]) != tuple(replacement_mask.shape):
            raise ValueError(f"Cached mask shape {replacement_mask.shape} does not match target ROI shape {target_roi.shape[:2]}.")

        target_roi[replacement_mask] = segment_data["replacement_color"]

    return target_rgb
