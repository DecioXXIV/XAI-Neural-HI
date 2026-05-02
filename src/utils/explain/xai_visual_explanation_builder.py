import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

class XaiVisualExplanationBuilder:
    def __init__(self): pass
    
    def build_visual_explanation(self, page_name: str, page_xai_dir: str, segments: np.ndarray, scores: Dict[str, float]):
        img = Image.open(os.path.join(page_xai_dir, f"{page_name}_forexp.png")).convert("RGB")
        map_to_plot = np.array(segments, dtype=np.float32)
        for k in scores.keys(): map_to_plot[segments == int(k)] = scores[k]
        
        score_values = np.array(list(scores.values()))
        vmin, vmax = score_values.min(), score_values.max()
        cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
        cmap.set_bad(color='black')
        
        # Ensure the figure matches the resolution of the original image
        dpi = img.info.get('dpi', (100,))[0]  # Use image DPI if available, default to 100
        figsize = (img.width / dpi, img.height / dpi)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.axis("off")
        ax.imshow(img, alpha=1.0)  # Show original image in the background
        ax.imshow(map_to_plot, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.5)  # Overlay heatmap with transparency

        # Save the visual explanation figure without the color bar
        fig.savefig(os.path.join(page_xai_dir, f"{page_name}_visual_exp.png"), bbox_inches="tight", pad_inches=0)
        plt.close(fig)