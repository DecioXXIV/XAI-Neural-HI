import PIL, cv2, math
import numpy as np
from typing import Dict, List, Tuple
from PIL import Image

from src.utils.explain.segments_handler import SegmentsHandler

# ---------------------------------------------------------------------------
# Foreground detection parameters
# ---------------------------------------------------------------------------
BG_KERNEL_MULT_FACTOR = 0.045   # Fraction of the smaller image dimension used to compute the median-blur kernel size.
BG_KERNEL_MIN = 31              # Minimum allowed kernel size (must be odd) to guarantee a meaningful background estimate.
THRESHOLD_SCALE = 1.0           # Multiplicative scale applied to Otsu's threshold; values < 1 make detection more aggressive.
COLOR_WEIGHT = 0.25             # Weight of the chromatic (LAB) deviation score in the combined ink score.
SATURATION_WEIGHT = 0.0         # Weight of the HSV saturation score; disabled by default to avoid parchment noise.

# ---------------------------------------------------------------------------
# Component cleaning parameters
# ---------------------------------------------------------------------------
MIN_AREA = 4                        # Minimum pixel area for a connected component to be retained.
BORDER_MARGIN = 1                   # Components touching the image border within this many pixels are removed.
REMOVE_BORDER_COMPONENTS = True     # Whether to discard components that touch the image border.
REMOVE_TOP_HORIZONTAL = True        # Whether to discard long horizontal strokes near the top of the image (e.g. ruling lines).
TOP_LINE_Y_MAX = 60                 # Maximum y-coordinate (top of component) for the top-horizontal removal rule.
LONG_HORIZONTAL_MIN_WIDTH = 90      # Minimum width for a component to be considered a long horizontal stroke.
LONG_HORIZONTAL_MAX_HEIGHT = 20     # Maximum height for a component to be considered a long horizontal stroke.
LONG_HORIZONTAL_MIN_RATIO = 10.0    # Minimum width-to-height aspect ratio for a long horizontal stroke.
MAX_COMPONENT_AREA_FRACTION = 0.0   # Maximum allowed component area as a fraction of the image area (0 = disabled).

# ---------------------------------------------------------------------------
# Scale estimation parameters
# ---------------------------------------------------------------------------
MIN_CC_WIDTH = 1                    # Minimum width for a component to be included in scale estimation.
MIN_CC_HEIGHT = 2                   # Minimum height for a component to be included in scale estimation.
CC_WIDTH_MAX_SCALE_FACTOR = 0.35    # Components wider than this fraction of the image width are excluded from scale estimation.
CC_HEIGHT_MAX_SCALE_FACTOR = 0.25   # Components taller than this fraction of the image height are excluded from scale estimation.

# ---------------------------------------------------------------------------
# Grouping / segmentation parameters
# ---------------------------------------------------------------------------
DILATION_ITERATIONS = 1             # Number of dilation iterations applied when grouping by morphological expansion.
DILATION_KERNEL_SHAPE = "ellipse"   # Default structuring element shape for dilation grouping ("ellipse", "rect", or "cross").
HULL_MIN_CONTOUR_AREA = 0.0         # Minimum contour area to be included in convex-hull grouping.

# ---------------------------------------------------------------------------
# Character-level segmentation parameters
# ---------------------------------------------------------------------------
CHAR_VALLEY_REL_THRESHOLD = 0.16        # Relative height threshold for a valley in the vertical projection to be a cut candidate.
CHAR_MIN_COMPONENT_WIDTH_FACTOR = 1.75  # Minimum component width (multiple of typical_height) to attempt character splitting.
CHAR_MIN_PIECE_WIDTH_FACTOR = 0.28      # Minimum width of a resulting piece (multiple of typical_height) after a cut.
CHAR_MIN_CUT_DISTANCE_FACTOR = 0.42     # Minimum horizontal distance between two consecutive cuts (multiple of typical_height).
CHAR_SMOOTH_WINDOW = 5                  # Smoothing window (pixels) applied to the vertical projection before valley detection.
CHAR_CUT_WIDTH = 1                      # Width (pixels) of the vertical cut applied to separate character strokes.

def _auto_background_kernel(image_bgr: np.ndarray) -> int:
    """
    Compute an appropriate odd kernel size for median-blur background estimation.

    The kernel is proportional to the smaller image dimension so it adapts to
    image resolution.
    """
    height, width = image_bgr.shape[:2]

    # Scale the kernel to the smaller dimension to adapt to image resolution.
    kernel_size = int(min(height, width) * BG_KERNEL_MULT_FACTOR)

    if kernel_size < BG_KERNEL_MIN: return BG_KERNEL_MIN
    else:
        # cv2.medianBlur requires an odd kernel size
        if kernel_size % 2 == 0: kernel_size += 1
        return kernel_size


def _connected_component_label_mask(foreground_binary: np.ndarray) -> Tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """
    Run 8-connected component analysis on a binary foreground mask.

    Returns a tuple of (label_map, component_count, stats, centroids).
    The background label (0) is excluded from the returned component count.
    """
    binary = (foreground_binary > 0).astype(np.uint8)
    num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Subtract 1 from num_labels to exclude the background component (label 0).
    return label_map.astype(np.int32), int(num_labels - 1), stats, centroids

def _odd_at_least(value: int, minimum: int = 1) -> int:
    """
    Round value to the nearest integer, enforce a minimum, and ensure it is odd.

    Useful for computing kernel sizes that must satisfy both a lower bound and
    the odd-number constraint required by several OpenCV functions.
    """
    value = int(round(value))
    value = max(value, minimum)
    if value % 2 == 0:
        value += 1
    return value

def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    """
    Apply a 1-D moving (box) average to array x with the given window size.

    The window is forced to be odd and at least 1. Edge values are extended
    (mode="edge") to avoid boundary artefacts. Returns a float32 array of
    the same length as x.
    """
    window = _odd_at_least(window, minimum=1)

    # Short arrays or window=1 need no smoothing.
    if window <= 1 or len(x) < 3: return x.astype(np.float32)

    # Pad the signal on both sides by half the window to preserve output length.
    pad = window // 2
    x_padded = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    box_kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(x_padded, box_kernel, mode="valid")

def _project_proxy_labels_to_foreground(proxy_labels: np.ndarray, foreground_binary: np.ndarray) -> np.ndarray:
    """
    Restrict proxy_labels to foreground pixels and compact the label set.

    Pixels that are background in foreground_binary are forced to label 0.
    The surviving labels are then remapped to consecutive integers starting
    from 1 so that the label set has no gaps.
    """
    # Zero-out any label that falls outside the foreground mask.
    foreground_mask = (foreground_binary > 0).astype(np.int32)
    labels = (proxy_labels.astype(np.int32) * foreground_mask).astype(np.int32)

    # Collect the unique non-zero (foreground) labels.
    unique_fg_labels = np.unique(labels)
    unique_fg_labels = unique_fg_labels[unique_fg_labels != 0]

    if len(unique_fg_labels) == 0: return labels

    # Remap labels to consecutive integers to eliminate gaps caused by the masking.
    remapped = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(unique_fg_labels, start=1): remapped[labels == old_id] = new_id
    return remapped

def _labels_by_dilation_grouping(foreground_binary: np.ndarray, kernel_width: int, kernel_height: int, kernel_shape: str = DILATION_KERNEL_SHAPE) -> np.ndarray:
    """
    Group foreground pixels into segments by morphological dilation.

    The binary foreground mask is dilated with a structuring element of the
    given shape and size, which merges nearby strokes into a single connected
    region. 
    Connected components are then computed on the dilated mask and the
    resulting labels are projected back onto the original foreground pixels.
    """
    binary = (foreground_binary > 0).astype(np.uint8)

    # Kernel dimensions must be odd and at least 1 for OpenCV compatibility.
    kernel_width = _odd_at_least(max(1, kernel_width), minimum=1)
    kernel_height = _odd_at_least(max(1, kernel_height), minimum=1)

    # Select the structuring element shape.
    if kernel_shape == "rect": morph_shape = cv2.MORPH_RECT
    elif kernel_shape == "cross": morph_shape = cv2.MORPH_CROSS
    else: morph_shape = cv2.MORPH_ELLIPSE

    structuring_element = cv2.getStructuringElement(morph_shape, (kernel_width, kernel_height))

    # Dilate to bridge gaps between nearby strokes.
    if DILATION_ITERATIONS > 0: dilated = cv2.dilate(binary, structuring_element, DILATION_ITERATIONS)
    else: dilated = binary.copy()

    # Label connected components on the dilated proxy mask.
    _, proxy_labels = cv2.connectedComponents(dilated, connectivity=8)

    # Map the proxy labels back to the original (non-dilated) foreground pixels.
    return _project_proxy_labels_to_foreground(proxy_labels, foreground_binary)

def _labels_by_convex_hull_grouping(foreground_binary: np.ndarray, pad_x: int, pad_y: int) -> np.ndarray:
    """
    Group foreground pixels into segments using per-contour convex hulls.

    A convex hull is drawn for every external contour in the foreground mask,
    creating a filled proxy mask that merges overlapping or nearby strokes.
    An optional rectangular padding (pad_x, pad_y) is applied via dilation to
    further bridge small gaps between strokes before connected-component labelling.
    """
    binary = (foreground_binary > 0).astype(np.uint8)

    # Find all external contours in the foreground mask.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_mask = np.zeros_like(binary, dtype=np.uint8)

    # Draw the filled convex hull of each contour onto the proxy mask.
    for contour in contours:
        if float(cv2.contourArea(contour)) < HULL_MIN_CONTOUR_AREA: continue
        hull = cv2.convexHull(contour)
        cv2.drawContours(hull_mask, [hull], -1, 1, thickness=cv2.FILLED)

    pad_x, pad_y = max(0, int(pad_x)), max(0, int(pad_y))

    # Optionally expand hull regions to close small inter-stroke gaps.
    if pad_x > 0 or pad_y > 0:
        pad_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * pad_x + 1, 2 * pad_y + 1))
        hull_mask = cv2.dilate(hull_mask, pad_kernel, iterations=1)

    # Label connected regions in the padded hull mask.
    _, proxy_labels = cv2.connectedComponents(hull_mask, connectivity=8)

    # Project the proxy labels back onto the actual foreground pixels.
    return _project_proxy_labels_to_foreground(proxy_labels, foreground_binary)

def _choose_vertical_cut_positions(roi_binary: np.ndarray, typical_height: float, valley_rel_threshold: float) -> List[int]:
    """
    Identify x-positions at which to cut a component ROI to separate characters.

    The function computes the vertical projection (column-wise foreground pixel
    count) of the binary ROI, smooths it, and finds columns where the projection
    drops below a relative threshold — indicating a gap between character strokes.
    Cuts too close to the edges or to each other are discarded; when two minima
    are too close only the deeper one is retained.
    """
    h, w = roi_binary.shape
    if w < 3: return []

    # Compute and smooth the vertical projection (foreground pixel count per column).
    projection = roi_binary.sum(axis=0).astype(np.float32)
    smoothed_projection = _moving_average_1d(projection, CHAR_SMOOTH_WINDOW)

    # A column is a cut candidate if its smoothed count falls below the threshold.
    cut_threshold = max(1.0, float(valley_rel_threshold) * float(h))
    is_candidate = smoothed_projection <= cut_threshold

    # Exclude columns too close to the left or right edge — cutting there would
    # produce pieces too narrow to represent a single character.
    min_piece_width = max(2, int(round(CHAR_MIN_PIECE_WIDTH_FACTOR * typical_height)))
    min_cut_distance = max(2, int(round(CHAR_MIN_CUT_DISTANCE_FACTOR * typical_height)))
    is_candidate[:min_piece_width] = False
    is_candidate[max(0, w - min_piece_width):] = False

    # Detect contiguous candidate runs and pick the deepest valley in each run.
    cuts: List[int] = []
    run_start = None

    for col_idx, is_valley in enumerate(is_candidate):
        if is_valley and run_start is None:
            # Start of a new candidate run.
            run_start = col_idx
        elif not is_valley and run_start is not None:
            # End of a candidate run: keep the column with the lowest projection value.
            run_cols = np.arange(run_start, col_idx)
            cuts.append(int(run_cols[np.argmin(smoothed_projection[run_cols])]))
            run_start = None

    # Handle a candidate run that extends to the last column.
    if run_start is not None:
        run_cols = np.arange(run_start, w)
        cuts.append(int(run_cols[np.argmin(smoothed_projection[run_cols])]))

    # Filter cuts that are too close to each other, keeping the deeper one.
    filtered_cuts, last_cut = [], None
    for cut in cuts:
        if last_cut is None:
            filtered_cuts.append(cut)
            last_cut = cut
            continue

        if cut - last_cut >= min_cut_distance:
            # Cuts are far enough apart; keep both.
            filtered_cuts.append(cut)
            last_cut = cut
        else:
            # Cuts are too close: replace the previous one if the new valley is deeper.
            if smoothed_projection[cut] < smoothed_projection[filtered_cuts[-1]]:
                filtered_cuts[-1] = cut
                last_cut = cut

    return filtered_cuts

def _assign_removed_foreground_to_nearest_label(labels: np.ndarray, original_binary: np.ndarray) -> np.ndarray:
    """
    Reassign foreground pixels that lost their label during cutting to the nearest labelled component.

    After vertical cuts are applied, some foreground pixels may end up with
    label 0 (they were erased by the cut strip). This function uses a distance
    transform to find the nearest already-labelled foreground pixel for each
    such orphaned pixel and assigns it that neighbour's label.
    """
    original_foreground = (original_binary > 0)
    labels = labels.astype(np.int32).copy()

    # Identify pixels that are foreground in the original mask but carry no label.
    orphaned_pixels = original_foreground & (labels == 0)

    if not np.any(orphaned_pixels): return labels
    
    # No labelled pixels exist at all; nothing to propagate.
    if np.max(labels) == 0: return labels

    labelled_pixels = labels > 0

    # Build a source image for distanceTransformWithLabels:
    #   labelled pixels  →  0   (they are the "seeds")
    #   everything else  →  255 (distance is computed from these)
    distance_src = np.where(labelled_pixels, 0, 255).astype(np.uint8)

    # zone_labels encodes, for each non-seed pixel, which connected region of
    # seeds (Voronoi zone) is closest to it.
    _, zone_labels = cv2.distanceTransformWithLabels(distance_src, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_CCOMP)

    # Build a mapping: zone_id → most-frequent original label within that zone.
    # This tells us which segment label to propagate to pixels inside each zone.
    zone_to_label: Dict[int, int] = {}
    for zone_id in np.unique(zone_labels[labelled_pixels]):
        # Collect all non-zero labels that fall in this Voronoi zone.
        zone_pixel_labels = labels[labelled_pixels & (zone_labels == zone_id)]
        zone_pixel_labels = zone_pixel_labels[zone_pixel_labels > 0]
        if zone_pixel_labels.size == 0: continue
        # Majority vote: pick the most frequent label in the zone.
        label_counts = np.bincount(zone_pixel_labels.astype(np.int64))
        zone_to_label[int(zone_id)] = int(np.argmax(label_counts))

    if zone_to_label:
        # Convert the zone→label dict to a flat lookup array for vectorised assignment.
        max_zone_id = int(zone_labels.max())
        label_lookup = np.zeros(max_zone_id + 1, dtype=np.int32)
        for zone_id, original_label in zone_to_label.items():
            if 0 <= zone_id <= max_zone_id:
                label_lookup[zone_id] = int(original_label)

        # Look up the label for every orphaned pixel via its zone id.
        orphaned_zone_ids = zone_labels[orphaned_pixels]
        valid_zone_mask = (orphaned_zone_ids >= 0) & (orphaned_zone_ids <= max_zone_id)
        assigned_labels = np.zeros_like(orphaned_zone_ids, dtype=np.int32)
        assigned_labels[valid_zone_mask] = label_lookup[orphaned_zone_ids[valid_zone_mask]]

        # Write the assigned labels back into the label map.
        rows, cols = np.where(orphaned_pixels)
        labels[rows, cols] = assigned_labels

    return labels

def _labels_by_character_heuristic(foreground_binary: np.ndarray, raw_labels: np.ndarray, raw_stats: np.ndarray, typical_height: float, aggressiveness: float) -> np.ndarray:
    """
    Split connected components into individual character segments via vertical cuts.

    For each component that is wide enough (relative to the typical character
    height), the function analyses the vertical projection of the component ROI
    and erases thin vertical strips at valley positions to separate touching
    characters. Pixels removed by the cuts are then reassigned to the nearest
    surviving segment via nearest-label propagation.

    The aggressiveness parameter scales both the valley threshold (easier to find
    valleys at higher values) and the minimum component width required to attempt
    splitting (narrower components are also split at higher values).
    """
    binary = (foreground_binary > 0).astype(np.uint8)
    aggressiveness = max(0.1, float(aggressiveness))

    # Scale the valley threshold and minimum component width by aggressiveness.
    valley_rel_threshold = CHAR_VALLEY_REL_THRESHOLD * math.sqrt(aggressiveness)
    min_component_width_factor = CHAR_MIN_COMPONENT_WIDTH_FACTOR / math.sqrt(aggressiveness)
    min_component_width = max(4, int(round(min_component_width_factor * typical_height)))

    segmented = binary.copy()

    # Iterate over every connected component and attempt to cut wide ones.
    for label_id in range(1, raw_stats.shape[0]):
        x = int(raw_stats[label_id, cv2.CC_STAT_LEFT])
        y = int(raw_stats[label_id, cv2.CC_STAT_TOP])
        w = int(raw_stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(raw_stats[label_id, cv2.CC_STAT_HEIGHT])
        area = int(raw_stats[label_id, cv2.CC_STAT_AREA])

        # Skip components that are too small to contain more than one character.
        if area < 4 or w < min_component_width or h < 3: continue

        # Extract the binary ROI for this component only.
        component_roi = (raw_labels[y:y + h, x:x + w] == label_id).astype(np.uint8)

        # Find vertical cut positions within the ROI.
        cut_positions = _choose_vertical_cut_positions(component_roi, typical_height, valley_rel_threshold)

        # Erase a thin vertical strip at each cut position to separate the strokes.
        for cut_col in cut_positions:
            strip_left = max(0, cut_col - CHAR_CUT_WIDTH // 2)
            strip_right = min(w, strip_left + CHAR_CUT_WIDTH)
            segmented[y:y + h, x + strip_left:x + strip_right] = 0

    # Re-label connected components after all cuts have been applied.
    _, cut_labels = cv2.connectedComponents(segmented.astype(np.uint8), connectivity=8)
    cut_labels = cut_labels.astype(np.int32)

    # Restore foreground pixels erased by the cuts and assign them to the nearest component.
    labels = _assign_removed_foreground_to_nearest_label(cut_labels, binary)

    # Compact the label set so that labels are consecutive integers starting from 1.
    unique_fg_labels = np.unique(labels[labels > 0])
    remapped = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(unique_fg_labels, start=1):
        remapped[labels == old_id] = new_id

    return remapped

def _compute_grouping_parameters(granularity: str, aggressiveness: float, scale_info: Dict[str, float]) -> Dict[str, int]:
    """
    Derive dilation/hull padding parameters scaled to the typical text component size.

    For "word" granularity, horizontal padding is larger than vertical padding
    to bridge intra-word gaps while preserving inter-line separation. All values
    scale with aggressiveness so that higher values merge more strokes.
    For "custom" granularity, neutral (no-op) parameters are returned.
    """
    typical_h = max(3.0, float(scale_info["typical_component_height"]))
    aggressiveness = max(0.0, float(aggressiveness))

    if granularity == "word":
        # Horizontal padding: bridges gaps between letters within a word.
        word_pad_x = max(0, int(round(0.25 * typical_h * aggressiveness)))
        # Vertical padding: kept small to avoid merging adjacent text lines.
        word_pad_y = max(0, int(round(0.04 * typical_h * aggressiveness)))
        # Dilation kernel dimensions for the fallback dilation method.
        kernel_width = _odd_at_least(max(5, round(0.75 * typical_h * aggressiveness)), minimum=5)
        kernel_height = _odd_at_least(max(3, round(0.12 * typical_h * aggressiveness)), minimum=3)
        return {"kernel_width": kernel_width, "kernel_height": kernel_height, "word_pad_x": word_pad_x, "word_pad_y": word_pad_y}

    # "custom" granularity: return neutral parameters (no grouping applied).
    return {"kernel_width": 1, "kernel_height": 1, "word_pad_x": 0, "word_pad_y": 0}

class InkBasedSegmentsHandler(SegmentsHandler):
    def __init__(self, aggressiveness: float, granularity: str, method: str):
        super().__init__()
        self.aggressiveness = aggressiveness
        self.granularity = granularity
        self.method = method

    def generate_segments(self, padded_img: PIL.Image.Image) -> np.ndarray:
        """
        Full segmentation pipeline: foreground detection → cleaning → labelling.

        Converts the input PIL image to BGR, estimates the ink foreground,
        removes spurious components, analyses the text scale, and finally
        builds the segment label map according to the configured granularity
        and method.
        """
        img_rgb = np.array(padded_img)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Step 1: estimate which pixels belong to ink strokes.
        foreground_mask = self._estimate_foreground_binary(img_bgr)

        # Step 2: remove noise, border artefacts, and ruling lines.
        cleaned_foreground_mask = self._clean_foreground_components(foreground_mask)

        # Step 3: label individual strokes and derive typographic scale metrics.
        raw_labels, raw_count, raw_stats, _ = _connected_component_label_mask(cleaned_foreground_mask)
        scale_info = self._estimate_text_scale_from_components(raw_stats, cleaned_foreground_mask)

        # Step 4: group/split strokes into the desired segment granularity.
        labels = self._build_wise_labels(cleaned_foreground_mask, raw_labels, raw_count, raw_stats, scale_info)
        return labels.astype(np.int32)
    
    def create_segments_overlay(self, segments: np.ndarray, page_name: str) -> PIL.Image.Image:
        """
        Build an RGB visualization of the ink-based segment labels.

        Label 0 is rendered as black. Positive integer labels are mapped to
        deterministic high-contrast colours so repeated runs are comparable.
        """
        if segments.ndim != 2: raise ValueError(f"Segments map for '{page_name}' must be a 2-D array.")

        segment_ids = segments.astype(np.int64, copy=False)
        positive_mask = segment_ids > 0

        hsv_overlay = np.zeros((*segment_ids.shape, 3), dtype=np.uint8)
        hsv_overlay[..., 0] = ((segment_ids * 37) % 180).astype(np.uint8)
        hsv_overlay[..., 1] = np.where(positive_mask, 180 + ((segment_ids * 29) % 76), 0).astype(np.uint8)
        hsv_overlay[..., 2] = np.where(positive_mask, 220 + ((segment_ids * 17) % 36), 0).astype(np.uint8)

        rgb_overlay = cv2.cvtColor(hsv_overlay, cv2.COLOR_HSV2RGB)
        return Image.fromarray(rgb_overlay)
        
    def _estimate_foreground_binary(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Compute a binary foreground mask that identifies ink pixels.

        Three complementary cues are combined into a single ink score:
          1. Dark-ink score: how much darker each pixel is than its local background.
          2. Chromatic score: colour deviation from the local background in LAB space,
             which helps detect red or blue inks that are not much darker than parchment.
          3. Saturation score: absolute HSV saturation; useful for vivid inks but
             disabled by default to avoid false positives on discoloured parchment.

        Otsu's method is then applied to the combined score to find a global threshold.
        """
        bg_kernel = _auto_background_kernel(image_bgr)

        # ---- Dark-ink score ------------------------------------------------
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        bg_gray = cv2.medianBlur(gray, bg_kernel)
        # cv2.subtract clips at 0, so the result is positive only where the pixel
        # is darker than its local background.
        dark_score = cv2.subtract(bg_gray, gray).astype(np.float32)

        # ---- Chromatic score -----------------------------------------------
        # Work in LAB to measure perceptual colour difference from the local background.
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        lab_bg = cv2.medianBlur(lab, bg_kernel)
        delta_a = lab[:, :, 1].astype(np.float32) - lab_bg[:, :, 1].astype(np.float32)
        delta_b = lab[:, :, 2].astype(np.float32) - lab_bg[:, :, 2].astype(np.float32)
        chroma_delta = np.sqrt(delta_a * delta_a + delta_b * delta_b)

        # ---- Saturation score ----------------------------------------------
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)

        # ---- Combined ink score and thresholding ---------------------------
        ink_score = dark_score + COLOR_WEIGHT * chroma_delta + SATURATION_WEIGHT * saturation
        ink_score_u8 = cv2.normalize(ink_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Determine threshold via Otsu, then optionally scale it.
        otsu_threshold, _ = cv2.threshold(ink_score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_threshold = max(0.0, min(255.0, float(otsu_threshold) * THRESHOLD_SCALE))
        foreground_mask = np.where(ink_score_u8 > final_threshold, 255, 0).astype(np.uint8)

        return foreground_mask
    
    def _clean_foreground_components(self, foreground_mask: np.ndarray) -> np.ndarray:
        """
        Remove spurious connected components from the foreground mask.

        The following classes of components are discarded:
          - Components below the minimum pixel area (noise).
          - Components touching the image border (scan artefacts, frame lines).
          - Long, thin horizontal strokes near the top of the image (ruling lines).
          - Components that exceed the maximum allowed area fraction (large blobs).

        All retained components are written into a clean output mask.
        """
        binary = (foreground_mask > 0).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        height, width = binary.shape
        image_area = height * width
        cleaned = np.zeros_like(binary, dtype=np.uint8)

        for label_id in range(1, num_labels):
            x = int(stats[label_id, cv2.CC_STAT_LEFT])
            y = int(stats[label_id, cv2.CC_STAT_TOP])
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])
            aspect_ratio = w / max(h, 1)

            # Evaluate each rejection criterion.
            too_small = area < int(MIN_AREA)
            touches_border = (
                x <= BORDER_MARGIN
                or y <= BORDER_MARGIN
                or x + w >= width - BORDER_MARGIN
                or y + h >= height - BORDER_MARGIN
            )
            long_top_horizontal = (
                y < TOP_LINE_Y_MAX
                and h <= LONG_HORIZONTAL_MAX_HEIGHT
                and w >= LONG_HORIZONTAL_MIN_WIDTH
                and aspect_ratio >= LONG_HORIZONTAL_MIN_RATIO
            )
            too_large = (
                MAX_COMPONENT_AREA_FRACTION > 0
                and area > MAX_COMPONENT_AREA_FRACTION * image_area
            )

            if too_small: continue
            if REMOVE_BORDER_COMPONENTS and touches_border: continue
            if REMOVE_TOP_HORIZONTAL and long_top_horizontal: continue
            if too_large: continue

            cleaned[labels == label_id] = 255

        return cleaned.astype(np.uint8)
    
    def _estimate_text_scale_from_components(self, stats: np.ndarray, foreground_mask: np.ndarray) -> Dict[str, float]:
        """
        Estimate typical text component size from connected component statistics.

        Only components within plausible size bounds (not too tiny, not spanning
        most of the image) are included so that noise and large blobs do not
        skew the estimate. The median is used instead of the mean for robustness
        against outliers. Falls back to conservative defaults when no valid
        components are found.
        """
        height, width = foreground_mask.shape
        heights, widths, areas = [], [], []

        for label_id in range(1, stats.shape[0]):
            w = int(stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_id, cv2.CC_STAT_AREA])

            # Exclude components that are clearly noise or anomalously large.
            if area < MIN_AREA: continue
            if h < MIN_CC_HEIGHT or w < MIN_CC_WIDTH: continue
            if h > CC_HEIGHT_MAX_SCALE_FACTOR * height or w > CC_WIDTH_MAX_SCALE_FACTOR * width: continue

            heights.append(h)
            widths.append(w)
            areas.append(area)

        if not heights:
            # No valid components found; return safe fallback values.
            return {"typical_component_height": 10.0, "typical_component_width": 10.0, "typical_component_area": 50.0}

        return {
            "typical_component_height": float(np.median(np.asarray(heights, dtype=np.float32))),
            "typical_component_width":  float(np.median(np.asarray(widths,  dtype=np.float32))),
            "typical_component_area":   float(np.median(np.asarray(areas,   dtype=np.float32))),
        }

    def _build_wise_labels(self, foreground_binary: np.ndarray, raw_labels: np.ndarray, raw_count: int, raw_stats: np.ndarray, scale_info: Dict[str, float]) -> np.ndarray:
        """
        Dispatch to the appropriate labelling strategy based on granularity and method.

        Supported granularity levels:
          - "raw":    return raw connected-component labels without any grouping.
          - "char":   split components at character boundaries using the vertical-
                      projection heuristic.
          - "word":   merge components into word-level groups via hull or dilation.
          - "custom": apply user-specified grouping with neutral default parameters.
        """
        # Raw granularity: return connected-component labels as-is.
        if self.granularity == "raw": return raw_labels.astype(np.int32)

        # Character granularity: attempt to split touching characters.
        if self.granularity == "char":
            typical_height = float(scale_info["typical_component_height"])
            return _labels_by_character_heuristic(foreground_binary, raw_labels, raw_stats, typical_height, self.aggressiveness)

        # For word/custom granularity, first compute scale-aware parameters.
        params = _compute_grouping_parameters(self.granularity, self.aggressiveness, scale_info)

        if self.granularity == "word":
            if self.method in ("auto", "hull"):
                # Convex-hull method: merge strokes whose hulls overlap after padding.
                pad_x, pad_y = params["word_pad_x"], params["word_pad_y"]
                return _labels_by_convex_hull_grouping(foreground_binary, pad_x, pad_y)

            # Dilation method: merge strokes that touch after morphological expansion.
            kernel_width, kernel_height, kernel_shape = params["kernel_width"], params["kernel_height"], "rect"
            return _labels_by_dilation_grouping(foreground_binary, kernel_width, kernel_height, kernel_shape)

        if self.granularity == "custom":
            if self.method == "hull":
                pad_x, pad_y = params["word_pad_x"], params["word_pad_y"]
                return _labels_by_convex_hull_grouping(foreground_binary, pad_x, pad_y)

            kernel_width, kernel_height = params["kernel_width"], params["kernel_height"]
            return _labels_by_dilation_grouping(foreground_binary, kernel_width, kernel_height)