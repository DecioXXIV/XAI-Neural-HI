import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from typing import Any, Dict, List, Tuple

from src.utils.constants import EXPERIMENTS_ROOT
from src.utils.faithfulness.faithfulness_figures_retriever import FaithfulnessFiguresRetriever
from src.utils.faithfulness.faithfulness_figures_utils import get_masking_xlabel, get_xai_plot_style
from src.utils.logger import Logger

logger = Logger()

LINEWIDTH_MAIN = 2.6
LINEWIDTH_BASE = 1.6
MARKERSIZE = 5
MARKEREDGEWIDTH = 1.2
RANDOM_ALPHA = 0.78
SALIENCY_ZORDER = 4
RANDOM_ZORDER = 3
GRID_ALPHA = 0.25
BAND_ALPHA = 0.33
BOX_ALPHA = 0.85
BOX_LINEWIDTH = 0.9
CENTRAL_LINEWIDTH = 0.8
ACC_YMIN = -0.03
ACC_YMAX = 1.03
AREA_YMIN = -0.03
AREA_YMAX = 1.03

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
})


class FaithfulnessFiguresGenerator:
    def __init__(self, experiment_id: str, output_xai_entry: str, xai_pairs: List[Tuple[str, str]], mask_ceil: float, mask_step: float, patches_colors: List[str]):
        self.experiment_id = experiment_id
        self.output_xai_entry = output_xai_entry
        self.xai_pairs = xai_pairs
        self.mask_ceil = float(mask_ceil)
        self.mask_step = float(mask_step)
        self.patches_colors = patches_colors
        self.data_retriever = FaithfulnessFiguresRetriever(experiment_id, self.mask_ceil, self.mask_step)
        self.output_dir = os.path.join(EXPERIMENTS_ROOT, self.experiment_id, "output", "faithfulness_figures", self.output_xai_entry)

    def __call__(self):
        logger.info(f"*** BEGINNING OF FAITHFULNESS FIGURES GENERATION -> Experiment: {self.experiment_id} | Output XAI Entry: {self.output_xai_entry} ***")
        xai_data_by_color = {patches_color: self._retrieve_xai_data(patches_color) for patches_color in self.patches_colors}

        os.makedirs(self.output_dir, exist_ok=True)
        for patches_color, xai_data in xai_data_by_color.items():
            self._generate_figure(xai_data, patches_color)

        logger.info(f"*** FAITHFULNESS FIGURES GENERATED SUCCESSFULLY -> Experiment: {self.experiment_id} | Output Directory: {self.output_dir} ***")

    def _retrieve_xai_data(self, patches_color: str) -> List[Dict[str, Any]]:
        xai_data = []

        for xai_algorithm, xai_entry in self.xai_pairs:
            faithfulness_data = self.data_retriever.get_faithfulness_data(xai_algorithm, xai_entry, patches_color)
            faithfulness_data["xai_algorithm"] = xai_algorithm
            faithfulness_data["xai_entry"] = xai_entry
            faithfulness_data["masked_area_data"] = self.data_retriever.get_masked_area_data(xai_algorithm, xai_entry, patches_color)
            xai_data.append(faithfulness_data)

        self._validate_xai_data(xai_data, patches_color)
        return xai_data

    def _validate_xai_data(self, xai_data: List[Dict[str, Any]], patches_color: str):
        reference_rates = xai_data[0]["rates"]

        for data in xai_data[1:]:
            if not np.allclose(data["rates"], reference_rates):
                raise ValueError(f"Mask rates for '{data['xai_algorithm']}/{data['xai_entry']}' do not match the other configured pairs for color '{patches_color}'.")

    def _generate_figure(self, xai_data: List[Dict[str, Any]], patches_color: str):
        rates = xai_data[0]["rates"]
        fig, (ax_acc, ax_area) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.08})
        fig.suptitle(f"{self.experiment_id} -- {self.output_xai_entry} -- {patches_color} patches")

        self._plot_accuracy(ax_acc, rates, xai_data)
        self._plot_masked_area(ax_area, rates, xai_data, patches_color)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_path = os.path.join(self.output_dir, self._get_output_filename(patches_color))
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Faithfulness figure saved to '{output_path}'.")

    def _plot_accuracy(self, ax, rates: np.ndarray, xai_data: List[Dict[str, Any]]):
        relevance_handles, random_handles = [], []

        for data in xai_data:
            style = get_xai_plot_style(data["xai_algorithm"])
            color, label = style["color"], style["label"]

            relevance_handle, = ax.plot(rates, data["saliency_accuracies"], color=color, linestyle="-", marker="o", linewidth=LINEWIDTH_MAIN, markersize=MARKERSIZE, markerfacecolor=color, markeredgecolor=color, markeredgewidth=MARKEREDGEWIDTH, zorder=SALIENCY_ZORDER, label=f"{label} (saliency)")
            random_handle, = ax.plot(rates, data["random_mean_accuracies"], color=color, linestyle=style["random_linestyle"], marker="D", linewidth=LINEWIDTH_BASE, markersize=MARKERSIZE, markerfacecolor="white", markeredgecolor=color, markeredgewidth=MARKEREDGEWIDTH, alpha=RANDOM_ALPHA, zorder=RANDOM_ZORDER, label=f"{label} (random)")

            if data["n_random_runs"] > 1:
                ax.fill_between(rates, data["random_min_accuracies"], data["random_max_accuracies"], color=color, alpha=BAND_ALPHA, zorder=1, label="_nolegend_")

            relevance_handles.append(relevance_handle)
            random_handles.append(random_handle)

        ax.set_xlim(self._get_xmin(rates), self._get_xmax(rates))
        ax.set_ylim(ACC_YMIN, ACC_YMAX)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("(Crop-level) Test Accuracy")
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA)
        ax.tick_params(axis="x", which="both", labelbottom=False)
        self._build_accuracy_legend(ax, relevance_handles, random_handles)

    def _plot_masked_area(self, ax, rates: np.ndarray, xai_data: List[Dict[str, Any]], patches_color: str):
        positions, box_width = self._get_boxplot_positions(rates, len(xai_data))

        for data, method_positions in zip(xai_data, positions):
            style = get_xai_plot_style(data["xai_algorithm"])
            self._draw_boxplot(ax, data["masked_area_data"], method_positions, box_width, style["color"])

        ax.set_xlim(self._get_xmin(rates), self._get_xmax(rates))
        ax.set_ylim(AREA_YMIN, AREA_YMAX)
        ax.set_xticks(rates)
        ax.set_yticks(np.arange(0.0, 1.01, 0.1))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.set_ylabel("Masked Page Area")
        ax.set_xlabel(get_masking_xlabel(patches_color))
        ax.grid(True, linestyle="--", alpha=GRID_ALPHA, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        self._build_area_legend(ax, xai_data)

    def _get_boxplot_positions(self, rates: np.ndarray, n_methods: int) -> Tuple[List[np.ndarray], float]:
        min_step = float(np.min(np.diff(rates)))
        total_group_width = min_step * 0.84
        subgroup_width = total_group_width / n_methods
        box_width = subgroup_width * 0.72
        offsets = (np.arange(n_methods) - (n_methods - 1) / 2.0) * subgroup_width
        return [rates + offset for offset in offsets], box_width

    def _draw_boxplot(self, ax, data_by_rate: List[np.ndarray], positions: np.ndarray, width: float, color: str):
        boxplot = ax.boxplot(data_by_rate, positions=positions, widths=width, patch_artist=True, whis=(0, 100), showfliers=False, showmeans=True, meanline=True, manage_ticks=False)

        for box in boxplot["boxes"]:
            box.set(facecolor=color, edgecolor=color, alpha=BOX_ALPHA, linewidth=0.0)
        for whisker in boxplot["whiskers"]:
            whisker.set(color=color, linewidth=BOX_LINEWIDTH)
        for cap in boxplot["caps"]:
            cap.set(color=color, linewidth=BOX_LINEWIDTH)
        for median in boxplot["medians"]:
            median.set(color=(0, 0, 0, 0), linewidth=0.0)
        for mean in boxplot["means"]:
            mean.set(color="black", linewidth=CENTRAL_LINEWIDTH, linestyle="-")

    def _build_accuracy_legend(self, ax, relevance_handles, random_handles):
        legend_handles = relevance_handles + random_handles
        ax.legend(legend_handles, [handle.get_label() for handle in legend_handles], loc="best", frameon=True, fontsize=8.0 if len(self.xai_pairs) > 2 else 8.5, ncol=2, handlelength=2.0, columnspacing=0.8, labelspacing=0.3, borderpad=0.3)

    def _build_area_legend(self, ax, xai_data: List[Dict[str, Any]]):
        handles = []

        for data in xai_data:
            style = get_xai_plot_style(data["xai_algorithm"])
            handles.append(Patch(facecolor=style["color"], edgecolor=style["color"], linewidth=0.0, alpha=BOX_ALPHA, label=style["label"]))

        ax.legend(handles=handles, loc="best", frameon=True, fontsize=8.5 if len(xai_data) > 2 else 9, ncol=len(handles), handlelength=2.0, columnspacing=0.8, labelspacing=0.3, borderpad=0.3)

    def _get_xmin(self, rates: np.ndarray) -> float:
        _, left_edges = self._get_boxplot_edges(rates)
        plot_xmin = min(float(np.min(rates)), float(np.min(left_edges)))
        return plot_xmin - self._get_x_padding(rates)

    def _get_xmax(self, rates: np.ndarray) -> float:
        right_edges, _ = self._get_boxplot_edges(rates)
        plot_xmax = max(float(np.max(rates)), float(np.max(right_edges)))
        return plot_xmax + self._get_x_padding(rates)

    def _get_boxplot_edges(self, rates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        positions, box_width = self._get_boxplot_positions(rates, len(self.xai_pairs))
        left_edges = np.concatenate([positions_ - box_width / 2.0 for positions_ in positions])
        right_edges = np.concatenate([positions_ + box_width / 2.0 for positions_ in positions])
        return right_edges, left_edges

    def _get_x_padding(self, rates: np.ndarray) -> float:
        return 0.03 * (float(np.max(rates)) - float(np.min(rates)))

    def _get_output_filename(self, patches_color: str) -> str:
        ceil_tag = str(self.mask_ceil).replace(".", "p")
        step_tag = str(self.mask_step).replace(".", "p")
        return f"combined_{patches_color}_ceil{ceil_tag}_step{step_tag}.png"
