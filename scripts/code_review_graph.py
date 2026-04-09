"""
Script: scripts/code_review_graph.py
Purpose: Generate a code review dependency graph for the Semantic-Divergence-of-DNN project.

Renders a publication-quality directed graph showing:
  - Module import dependencies (solid arrows)
  - Data flow between components (dashed arrows)
  - Execution layer hierarchy (top → bottom = setup → results)
  - Color coding by component type

Output: results/code_review_graph.png  (and .pdf if LaTeX is available)

Run from project root:
    python scripts/code_review_graph.py
"""

from __future__ import annotations

import ast
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────────────────────
# NODE DEFINITIONS
# Each node: id → {label, pos(x,y), color, layer_name, tooltip}
# x ∈ [0, 9], y ∈ [0, 7]  (y=7 = top)
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "tf_core"   : "#3498DB",   # blue  — TF shared library
    "tf_exp"    : "#85C1E9",   # light blue — TF experiments
    "vit_core"  : "#27AE60",   # green  — ViT / PyTorch core
    "vit_exp"   : "#82E0AA",   # light green — ViT experiments
    "setup"     : "#E67E22",   # orange — data setup
    "notebook"  : "#8E44AD",   # purple — Jupyter notebooks
    "core_contrib": "#E74C3C", # red    — paper's core contribution
    "validate"  : "#16A085",   # teal   — statistical validation
    "artifact"  : "#95A5A6",   # grey   — data artifacts (files/dirs)
}

NODES: dict[str, dict] = {
    # ── Layer 7: Raw datasets ──
    "caltech_zip": {
        "label": "caltech-101.zip\n(132 MB)",
        "pos": (1.0, 7.0), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Raw Data",
    },
    "archive_zip": {
        "label": "archive.zip\n(ImageNet-100, 17 GB)",
        "pos": (7.5, 7.0), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Raw Data",
    },

    # ── Layer 6: Data setup scripts ──
    "setup_caltech": {
        "label": "setup_caltech101.py",
        "pos": (1.0, 5.8), "color": COLORS["setup"], "shape": "rect",
        "layer": "Data Setup",
    },
    "setup_imagenet": {
        "label": "setup_imagenet100.py",
        "pos": (7.5, 5.8), "color": COLORS["setup"], "shape": "rect",
        "layer": "Data Setup",
    },

    # ── Layer 5: Extracted data artifacts ──
    "caltech_data": {
        "label": "caltech101_data/\n101_ObjectCategories/",
        "pos": (1.0, 4.6), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Data Artifacts",
    },
    "imagenet_data": {
        "label": "ImageNet100_Training/\ndata/{train,val,test}/",
        "pos": (7.5, 4.6), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Data Artifacts",
    },

    # ── Layer 4: Training ──
    "model_training": {
        "label": "Model_Training.ipynb\n(CNN training)",
        "pos": (1.0, 3.4), "color": COLORS["notebook"], "shape": "rect",
        "layer": "Training",
    },
    "checkpoints": {
        "label": "checkpoints/\n*.h5  (3 CNNs)",
        "pos": (2.8, 4.2), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Data Artifacts",
    },
    "frozen_split": {
        "label": "frozen_split_indices.json\nseed=42, 869 samples",
        "pos": (2.8, 3.0), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Data Artifacts",
    },
    "train_swin": {
        "label": "train_swin.py\n(Swin-Base, AMP)",
        "pos": (7.5, 3.4), "color": COLORS["vit_core"], "shape": "rect",
        "layer": "Training",
    },
    "swin_ckpt": {
        "label": "Swin_Training/\nswin_base_imagenet100.pt",
        "pos": (6.2, 2.4), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Data Artifacts",
    },

    # ── Layer 3: Core libraries ──
    "shared_utils": {
        "label": "src/shared_utils.py\n(TF · load · attacks · eval)",
        "pos": (3.8, 3.8), "color": COLORS["tf_core"], "shape": "rect",
        "layer": "Core Libraries",
    },
    "adv_bank_cls": {
        "label": "src/attacks/\nadversarial_bank.py",
        "pos": (3.8, 2.6), "color": COLORS["tf_core"], "shape": "rect",
        "layer": "Core Libraries",
    },
    "swin_utils": {
        "label": "VIT/src/swin_utils.py\n(PyTorch · HuggingFace)",
        "pos": (6.5, 3.8), "color": COLORS["vit_core"], "shape": "rect",
        "layer": "Core Libraries",
    },

    # ── Layer 2: Experiment scripts ──
    "build_bank": {
        "label": "build_adversarial_bank.py\n(Task 2)",
        "pos": (0.5, 1.8), "color": COLORS["tf_exp"], "shape": "rect",
        "layer": "Experiments",
    },
    "grad_mask": {
        "label": "gradient_masking_test.py\n(Task 1 ✓)",
        "pos": (2.2, 1.8), "color": COLORS["tf_exp"], "shape": "rect",
        "layer": "Experiments",
    },
    "transfer_mat": {
        "label": "transfer_attack_matrix.py\n(Task 3)",
        "pos": (3.8, 1.8), "color": COLORS["tf_exp"], "shape": "rect",
        "layer": "Experiments",
    },
    "confusion": {
        "label": "confusion_direction_analysis.py\n(Task 4 ★ CORE CONTRIBUTION)",
        "pos": (5.6, 1.8), "color": COLORS["core_contrib"], "shape": "rect",
        "layer": "Experiments",
    },
    "cnn_attacks_nb": {
        "label": "cnn-attacks.ipynb\n(original impl.)",
        "pos": (7.5, 1.8), "color": COLORS["notebook"], "shape": "rect",
        "layer": "Experiments",
    },

    # ── Layer 1: Validation ──
    "multi_seed": {
        "label": "multi_seed_runner.py\n(Task 5a · seeds 42/123/456)",
        "pos": (1.5, 0.6), "color": COLORS["validate"], "shape": "rect",
        "layer": "Validation",
    },
    "aggregate": {
        "label": "aggregate_results.py\n(Task 5b · bootstrap CI · Wilcoxon)",
        "pos": (4.0, 0.6), "color": COLORS["validate"], "shape": "rect",
        "layer": "Validation",
    },

    # ── Layer 0: Paper outputs ──
    "paper_results": {
        "label": "results/\n(tables · figures · JSON)",
        "pos": (4.5, -0.3), "color": COLORS["artifact"], "shape": "ellipse",
        "layer": "Outputs",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# EDGE DEFINITIONS
# (src_id, dst_id, style, label)
# style: "import" = solid blue, "data" = dashed grey
# ─────────────────────────────────────────────────────────────────────────────

EDGES: list[tuple[str, str, str, str]] = [
    # Data flow: raw → extracted
    ("caltech_zip",    "setup_caltech",   "data",   ""),
    ("archive_zip",    "setup_imagenet",  "data",   ""),
    ("setup_caltech",  "caltech_data",    "data",   ""),
    ("setup_imagenet", "imagenet_data",   "data",   ""),

    # Training
    ("caltech_data",   "model_training",  "data",   ""),
    ("model_training", "checkpoints",     "data",   "produces"),
    ("model_training", "frozen_split",    "data",   "produces"),
    ("imagenet_data",  "train_swin",      "data",   ""),
    ("train_swin",     "swin_ckpt",       "data",   "produces"),
    ("swin_utils",     "train_swin",      "import", "imports"),

    # Core libs wired to data
    ("checkpoints",    "shared_utils",    "data",   "loads"),
    ("frozen_split",   "shared_utils",    "data",   "loads"),
    ("caltech_data",   "shared_utils",    "data",   ""),
    ("imagenet_data",  "swin_utils",      "data",   ""),

    # Experiment imports from shared_utils
    ("shared_utils",   "build_bank",      "import", "imports"),
    ("shared_utils",   "grad_mask",       "import", "imports"),
    ("shared_utils",   "transfer_mat",    "import", "imports"),
    ("shared_utils",   "confusion",       "import", "imports"),
    ("shared_utils",   "multi_seed",      "import", "imports"),

    # adversarial_bank class
    ("adv_bank_cls",   "build_bank",      "import", "imports"),
    ("shared_utils",   "adv_bank_cls",    "import", ""),

    # Notebook lineage
    ("cnn_attacks_nb", "shared_utils",    "data",   "extracted to"),

    # Task dependencies (data flow between experiments)
    ("build_bank",     "transfer_mat",    "data",   "bank →"),
    ("build_bank",     "confusion",       "data",   "bank →"),

    # Validation
    ("shared_utils",   "aggregate",       "import", ""),
    ("multi_seed",     "aggregate",       "data",   "feeds"),

    # All experiments → paper results
    ("grad_mask",      "paper_results",   "data",   ""),
    ("transfer_mat",   "paper_results",   "data",   ""),
    ("confusion",      "paper_results",   "data",   ""),
    ("aggregate",      "paper_results",   "data",   ""),
    ("swin_ckpt",      "paper_results",   "data",   ""),
]

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

NODE_W = 1.55
NODE_H = 0.48
ELLIPSE_W = 1.55
ELLIPSE_H = 0.42


def _node_center(nid: str) -> tuple[float, float]:
    return NODES[nid]["pos"]


def _draw_node(ax: plt.Axes, nid: str, node: dict):
    x, y   = node["pos"]
    color  = node["color"]
    label  = node["label"]
    shape  = node.get("shape", "rect")

    if shape == "ellipse":
        ellipse = mpatches.Ellipse(
            (x, y), width=ELLIPSE_W, height=ELLIPSE_H,
            facecolor=color, edgecolor="white",
            linewidth=1.2, alpha=0.90, zorder=3,
        )
        ax.add_patch(ellipse)
    else:
        box = FancyBboxPatch(
            (x - NODE_W / 2, y - NODE_H / 2),
            NODE_W, NODE_H,
            boxstyle="round,pad=0.04",
            facecolor=color, edgecolor="white",
            linewidth=1.4, alpha=0.93, zorder=3,
        )
        ax.add_patch(box)

    fontsize = 5.6 if "\n" in label else 6.4
    ax.text(
        x, y, label,
        ha="center", va="center",
        fontsize=fontsize, color="white",
        fontweight="bold", zorder=4,
        multialignment="center",
    )


def _edge_offset(nid: str, outgoing: bool) -> tuple[float, float]:
    """Return the attachment point on a node boundary."""
    x, y = NODES[nid]["pos"]
    shape = NODES[nid].get("shape", "rect")
    hw = (ELLIPSE_W if shape == "ellipse" else NODE_W) / 2
    hh = (ELLIPSE_H if shape == "ellipse" else NODE_H) / 2
    return x, y + hh if outgoing else y - hh


def _draw_edge(ax: plt.Axes, src: str, dst: str, style: str, label: str):
    x1, y1 = _node_center(src)
    x2, y2 = _node_center(dst)

    # choose exit / entry sides based on relative positions
    dy = y2 - y1
    dx = x2 - x1

    if abs(dy) >= abs(dx):           # predominantly vertical
        y_off_src =  (ELLIPSE_H if NODES[src].get("shape") == "ellipse" else NODE_H) / 2
        y_off_dst =  (ELLIPSE_H if NODES[dst].get("shape") == "ellipse" else NODE_H) / 2
        p1 = (x1, y1 - y_off_src)    # bottom of src
        p2 = (x2, y2 + y_off_dst)    # top of dst
    else:                            # predominantly horizontal
        x_off_src = (ELLIPSE_W if NODES[src].get("shape") == "ellipse" else NODE_W) / 2
        x_off_dst = (ELLIPSE_W if NODES[dst].get("shape") == "ellipse" else NODE_W) / 2
        sign = 1 if dx > 0 else -1
        p1 = (x1 + sign * x_off_src, y1)
        p2 = (x2 - sign * x_off_dst, y2)

    is_import = (style == "import")
    color     = "#3498DB" if is_import else "#7F8C8D"
    lw        = 1.0 if is_import else 0.8
    ls        = "-" if is_import else (0, (4, 2))

    ax.annotate(
        "",
        xy=p2, xycoords="data",
        xytext=p1, textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=2,
    )

    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(
            mx, my, label,
            ha="center", va="center",
            fontsize=4.5, color=color,
            bbox=dict(boxstyle="round,pad=0.08", fc="white", ec="none", alpha=0.75),
            zorder=5,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PARSE ACTUAL IMPORTS  (static analysis of .py files)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_imports(path: Path) -> list[str]:
    """Return list of imported top-level module names from a Python file."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return []
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.append(node.module.split(".")[0])
    return mods


FILE_TO_NODE: dict[str, str] = {
    "src/shared_utils.py":                          "shared_utils",
    "src/attacks/adversarial_bank.py":              "adv_bank_cls",
    "experiments/build_adversarial_bank.py":        "build_bank",
    "experiments/gradient_masking_test.py":         "grad_mask",
    "experiments/transfer_attack_matrix.py":        "transfer_mat",
    "experiments/confusion_direction_analysis.py":  "confusion",
    "scripts/multi_seed_runner.py":                 "multi_seed",
    "scripts/aggregate_results.py":                 "aggregate",
    "scripts/setup_caltech101.py":                  "setup_caltech",
    "scripts/setup_imagenet100.py":                 "setup_imagenet",
    "VIT/src/swin_utils.py":                        "swin_utils",
    "VIT/train_swin.py":                            "train_swin",
}

INTERNAL_MOD_MAP: dict[str, str] = {
    "shared_utils":   "shared_utils",
    "swin_utils":     "swin_utils",
    "adversarial_bank": "adv_bank_cls",
    "src":            "shared_utils",
}


def build_import_table() -> dict[str, list[str]]:
    """Parse all tracked Python files and return {file_node: [imported_node]}."""
    table: dict[str, list[str]] = defaultdict(list)
    for rel_path, node_id in FILE_TO_NODE.items():
        full = PROJECT_ROOT / rel_path
        if not full.exists():
            continue
        for mod in _parse_imports(full):
            target = INTERNAL_MOD_MAP.get(mod)
            if target and target != node_id:
                table[node_id].append(target)
    return table


# ─────────────────────────────────────────────────────────────────────────────
# LAYER BAND ANNOTATIONS
# ─────────────────────────────────────────────────────────────────────────────

BANDS: list[tuple[float, float, str, str]] = [
    (6.75, 7.35,  "Raw Datasets",     "#F0F3F4"),
    (5.35, 6.55,  "Data Setup",       "#FEF9E7"),
    (4.00, 5.25,  "Data Artifacts",   "#F9F0FF"),
    (2.85, 3.95,  "Core Libraries",   "#EAF4FB"),
    (2.20, 2.75,  "Training",         "#E9F7EF"),
    (1.25, 2.15,  "Experiments",      "#FDEDEC"),
    (0.10, 1.20,  "Validation",       "#E8F8F5"),
    (-0.55, 0.05, "Paper Outputs",    "#F2F3F4"),
]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────────────

def render(out_path: Path):
    fig, ax = plt.subplots(figsize=(18, 14))
    ax.set_xlim(-0.8, 9.3)
    ax.set_ylim(-0.7, 7.7)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── background layer bands ──
    for y_lo, y_hi, band_label, band_color in BANDS:
        ax.add_patch(mpatches.FancyBboxPatch(
            (-0.75, y_lo), 10.0, y_hi - y_lo,
            boxstyle="round,pad=0.0",
            facecolor=band_color, edgecolor="#CCCCCC",
            linewidth=0.5, zorder=0, alpha=0.60,
        ))
        ax.text(
            -0.55, (y_lo + y_hi) / 2, band_label,
            ha="left", va="center",
            fontsize=6.0, color="#666666",
            fontweight="bold", rotation=0, zorder=1,
        )

    # ── edges (drawn before nodes so nodes sit on top) ──
    for src, dst, style, label in EDGES:
        _draw_edge(ax, src, dst, style, label)

    # ── nodes ──
    for nid, node in NODES.items():
        _draw_node(ax, nid, node)

    # ── title ──
    ax.set_title(
        "Code Review Graph — Semantic-Divergence-of-DNN\n"
        "Adversarial Robustness Analysis: CNN → ViT → VLM pipeline",
        fontsize=13, fontweight="bold", pad=12, color="#2C3E50",
    )

    # ── legend ──
    legend_items = [
        mpatches.Patch(color=COLORS["tf_core"],      label="TF core library"),
        mpatches.Patch(color=COLORS["tf_exp"],       label="TF experiment"),
        mpatches.Patch(color=COLORS["vit_core"],     label="ViT / PyTorch"),
        mpatches.Patch(color=COLORS["setup"],        label="Data setup"),
        mpatches.Patch(color=COLORS["notebook"],     label="Jupyter notebook"),
        mpatches.Patch(color=COLORS["core_contrib"], label="Core paper contribution"),
        mpatches.Patch(color=COLORS["validate"],     label="Statistical validation"),
        mpatches.Patch(color=COLORS["artifact"],     label="Data artifact / file"),
        mpatches.Patch(color="#3498DB", label="→ Import dependency"),
        mpatches.Patch(color="#7F8C8D", label="⇢ Data flow"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower right",
        fontsize=6.5,
        framealpha=0.92,
        ncol=2,
        title="Legend",
        title_fontsize=7.0,
    )

    # ── import stats annotation ──
    import_table = build_import_table()
    most_imported = sorted(
        {n for edges in import_table.values() for n in edges},
        key=lambda n: sum(n in v for v in import_table.values()),
        reverse=True,
    )
    dep_text = "Import fan-in (most depended on):\n" + "\n".join(
        f"  {n}: {sum(n in v for v in import_table.values())} dependents"
        for n in most_imported[:4]
    )
    ax.text(
        -0.6, -0.55, dep_text,
        ha="left", va="bottom",
        fontsize=6.0, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FDFEFE", ec="#BFC9CA", lw=0.7),
        family="monospace",
    )

    plt.tight_layout(pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"✓ Graph saved: {out_path}")

    # Optional PDF
    try:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        print(f"✓ PDF  saved: {pdf_path}")
    except Exception:
        pass

    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY
# ─────────────────────────────────────────────────────────────────────────────

def _print_adjacency():
    """Print a text adjacency list for quick review without matplotlib."""
    print("\n=== Import Adjacency List (static analysis) ===")
    table = build_import_table()
    for src in sorted(table):
        for dst in sorted(set(table[src])):
            print(f"  {src:40s}  →  {dst}")

    print("\n=== Edge List (graph definition) ===")
    for src, dst, style, label in EDGES:
        tag = "import" if style == "import" else "data  "
        print(f"  [{tag}]  {src:38s} → {dst}  {label}")

    node_in_degree: dict[str, int] = defaultdict(int)
    for _, dst, _, _ in EDGES:
        node_in_degree[dst] += 1
    print("\n=== Top nodes by in-degree ===")
    for nid, deg in sorted(node_in_degree.items(), key=lambda x: -x[1])[:8]:
        print(f"  {nid:40s}  {deg} incoming edges")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate code review graph")
    parser.add_argument(
        "--out", type=Path,
        default=PROJECT_ROOT / "results" / "code_review_graph.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--text", action="store_true",
        help="Print adjacency list to stdout (no matplotlib needed)",
    )
    args = parser.parse_args()

    _print_adjacency()

    if not args.text:
        render(args.out)
