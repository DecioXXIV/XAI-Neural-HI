import os, json, base64
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from PIL import Image
from html import escape
from matplotlib.colors import LinearSegmentedColormap


class XaiVisualExplanationBuilder:
    INTERACTIVE_HTML_VERSION = "spotlight-v6-external-assets"
    TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
    INTERACTIVE_HTML_TEMPLATE = "interactive_explanation.html"
    INTERACTIVE_CSS_TEMPLATE = "interactive_explanation.css"
    INTERACTIVE_JS_TEMPLATE = "interactive_explanation.js"

    def __init__(self):
        self.vmin, self.vmax = -1, 1

    def build_visual_explanation(self, page_name: str, page_xai_dir: str, segments: np.ndarray, scores: Dict[str, float], seg_type: str = None):
        img = Image.open(os.path.join(page_xai_dir, f"{page_name}_forexp.png")).convert("RGB")
        scores_by_segment = self._coerce_scores(scores)
        scores_by_segment = self._filter_scores_for_segments(segments, scores_by_segment, seg_type)
        map_to_plot = self._build_score_map(segments, scores_by_segment)

        self._build_static_visual_explanation(page_name, page_xai_dir, img, map_to_plot)
        self._build_interactive_visual_explanation(page_name, page_xai_dir, segments, scores_by_segment)

    def _coerce_scores(self, scores: Dict[str, float]) -> Dict[int, float]:
        return {int(k): float(v) for k, v in scores.items()}

    def _build_score_map(self, segments: np.ndarray, scores: Dict[int, float]) -> np.ndarray:
        map_to_plot = np.zeros(segments.shape, dtype=np.float32)
        for segment_id, score in scores.items():
            map_to_plot[segments == segment_id] = score
        return map_to_plot

    def _filter_scores_for_segments(self, segments: np.ndarray, scores: Dict[int, float], seg_type: str = None) -> Dict[int, float]:
        return {segment_id: score for segment_id, score in scores.items() if not self._is_background_segment_id(segments, segment_id, seg_type)}

    def _is_background_segment_id(self, segments: np.ndarray, segment_id: int, seg_type: str = None) -> bool:
        if int(segment_id) != 0 or segments.size == 0:
            return False
        if seg_type == "ink_based":
            return True
        return (np.count_nonzero(segments == 0) / segments.size) > 0.2

    def _build_static_visual_explanation(self, page_name: str, page_xai_dir: str, img: Image.Image, map_to_plot: np.ndarray):
        cmap = LinearSegmentedColormap.from_list("RdWhGn", ["red", "white", "green"])
        cmap.set_bad(color='black')

        # Plotting the attribution map
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis("off")
        ax.imshow(img, alpha=1.0)  # Show original image in the background
        ax.imshow(map_to_plot, cmap=cmap, vmin=self.vmin, vmax=self.vmax, alpha=0.5)  # Overlay heatmap with transparency

        # Save the visual explanation figure without the color bar
        fig.savefig(os.path.join(page_xai_dir, f"{page_name}_visual_exp.png"), bbox_inches="tight", dpi=300)
        plt.close(fig)

    def _build_interactive_visual_explanation(self, page_name: str, page_xai_dir: str, segments: np.ndarray, scores: Dict[int, float]):
        segment_map_path = os.path.join(page_xai_dir, f"{page_name}_segment_id_map.png")
        scores_path = os.path.join(page_xai_dir, f"{page_name}_interactive_scores.json")
        html_path = os.path.join(page_xai_dir, f"{page_name}_interactive.html")
        css_path = os.path.join(page_xai_dir, f"{page_name}_interactive.css")
        data_js_path = os.path.join(page_xai_dir, f"{page_name}_interactive_data.js")
        js_path = os.path.join(page_xai_dir, f"{page_name}_interactive.js")

        scores_json = {str(k): float(v) for k, v in sorted(scores.items())}

        self._save_segment_id_map(segments, segment_map_path)
        with open(scores_path, "w") as f:
            json.dump(scores_json, f, indent=4)
        with open(segment_map_path, "rb") as f:
            segment_map_data_uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")

        html = self._render_interactive_html(
            page_name=page_name,
            css_file_name=os.path.basename(css_path),
            data_js_file_name=os.path.basename(data_js_path),
            js_file_name=os.path.basename(js_path),
        )
        data_js = self._render_interactive_data_js(
            page_name=page_name,
            page_image_name=f"{page_name}_forexp.png",
            segment_map_src=segment_map_data_uri,
            scores=scores_json,
        )

        self._write_template_asset(self.INTERACTIVE_CSS_TEMPLATE, css_path)
        self._write_template_asset(self.INTERACTIVE_JS_TEMPLATE, js_path)
        with open(data_js_path, "w", encoding="utf-8") as f:
            f.write(data_js)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    def _save_segment_id_map(self, segments: np.ndarray, segment_map_path: str):
        segment_ids = np.asarray(segments, dtype=np.int64)
        min_id = int(segment_ids.min()) if segment_ids.size > 0 else 0
        max_id = int(segment_ids.max()) if segment_ids.size > 0 else 0
        if min_id < 0:
            raise ValueError("Segment ids must be non-negative to build the interactive visualization.")
        if max_id > 0xFFFFFF:
            raise ValueError("Segment ids exceed the 24-bit RGB encoding limit.")

        rgb_map = np.empty((*segment_ids.shape, 3), dtype=np.uint8)
        rgb_map[..., 0] = ((segment_ids >> 16) & 255).astype(np.uint8)
        rgb_map[..., 1] = ((segment_ids >> 8) & 255).astype(np.uint8)
        rgb_map[..., 2] = (segment_ids & 255).astype(np.uint8)
        Image.fromarray(rgb_map, "RGB").save(segment_map_path)

    @classmethod
    def interactive_visualization_is_current(cls, html_path: str) -> bool:
        if not os.path.exists(html_path):
            return False
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            return cls.INTERACTIVE_HTML_VERSION in f.read(2048)

    def _render_interactive_html(self, page_name: str, css_file_name: str, data_js_file_name: str, js_file_name: str) -> str:
        replacements = {
            "__XAI_INTERACTIVE_VERSION__": escape(self.INTERACTIVE_HTML_VERSION, quote=True),
            "__XAI_PAGE_TITLE__": escape(page_name),
            "__XAI_CSS_FILE__": escape(css_file_name, quote=True),
            "__XAI_DATA_FILE__": escape(data_js_file_name, quote=True),
            "__XAI_JS_FILE__": escape(js_file_name, quote=True),
        }
        html = self._load_template(self.INTERACTIVE_HTML_TEMPLATE)
        for placeholder, value in replacements.items():
            html = html.replace(placeholder, value)
        return html

    def _render_interactive_data_js(self, page_name: str, page_image_name: str, segment_map_src: str, scores: Dict[str, float]) -> str:
        payload = {
            "version": self.INTERACTIVE_HTML_VERSION,
            "pageName": page_name,
            "pageImageSrc": page_image_name,
            "segmentMapSrc": segment_map_src,
            "scores": scores,
        }
        payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        return f"window.XAI_INTERACTIVE_DATA = Object.freeze({payload_json});\n"

    def _load_template(self, template_name: str) -> str:
        with open(os.path.join(self.TEMPLATE_DIR, template_name), "r", encoding="utf-8") as f:
            return f.read()

    def _write_template_asset(self, template_name: str, output_path: str):
        with open(os.path.join(self.TEMPLATE_DIR, template_name), "r", encoding="utf-8") as f:
            content = f.read()
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
