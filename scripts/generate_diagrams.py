"""Generates publication-quality schematic diagrams for the project report and presentation."""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 6), dpi=200)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Colors
    c_data = "#1e3a8a"    # Blue
    c_model = "#047857"   # Emerald
    c_api = "#b45309"     # Amber
    c_ui = "#4338ca"      # Indigo

    # Data Tier Box
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.8), 2.2, 4.4, boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor=c_data, linewidth=2))
    ax.text(1.6, 5.0, "DATA LAYER\n(src/data)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_data)
    ax.text(1.6, 3.8, "• UrbanLayoutSample Contract\n• LayoutRasterizer (7 Channels)\n• Signed Distance Field (SDF)\n• GeometricFeatureExtractor\n• Zero-Leakage Splitter\n• Train-Fitted Normalizers", ha="center", va="center", fontsize=8.5, color="#1e293b", linespacing=1.6)

    # Models Tier Box
    ax.add_patch(patches.FancyBboxPatch((3.2, 0.8), 2.2, 4.4, boxstyle="round,pad=0.2", facecolor="#ecfdf5", edgecolor=c_model, linewidth=2))
    ax.text(4.3, 5.0, "MODEL LAYER\n(src/models)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_model)
    ax.text(4.3, 3.8, "• CompactUNet (ResBlocks)\n• CompactCNN (Dilated 2D)\n• RandomForestBaseline\n• Solid Outdoor Masking\n• NeuralSurrogateTrainer\n• Masked L1 + Grad Loss", ha="center", va="center", fontsize=8.5, color="#1e293b", linespacing=1.6)

    # API Tier Box
    ax.add_patch(patches.FancyBboxPatch((5.9, 0.8), 2.2, 4.4, boxstyle="round,pad=0.2", facecolor="#fffbeb", edgecolor=c_api, linewidth=2))
    ax.text(7.0, 5.0, "SERVICE & API\n(src/api)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_api)
    ax.text(7.0, 3.8, "• MicroclimateService Facade\n• FastAPI REST Router\n• Strict Provenance Schemas\n• Spatial Point Probe (X, Y)\n• 1D Canyon Transect\n• Dataset/CFD Inventories", ha="center", va="center", fontsize=8.5, color="#1e293b", linespacing=1.6)

    # UI Tier Box
    ax.add_patch(patches.FancyBboxPatch((8.6, 0.8), 1.9, 4.4, boxstyle="round,pad=0.2", facecolor="#eef2ff", edgecolor=c_ui, linewidth=2))
    ax.text(9.55, 5.0, "STUDIO UI\n(src/ui)", ha="center", va="center", fontsize=11, fontweight="bold", color=c_ui)
    ax.text(9.55, 3.8, "• HTML5 Canvas\n• Viridis/Magma\n• Particle Streamlines\n• Probe Crosshair\n• Interactive Transects\n• Triplet PNG Export", ha="center", va="center", fontsize=8.5, color="#1e293b", linespacing=1.6)

    # Connectors
    kw = dict(arrowstyle="->", color="#64748b", lw=2, mutation_scale=15)
    ax.annotate("", xy=(3.2, 3.0), xytext=(2.7, 3.0), arrowprops=kw)
    ax.annotate("", xy=(5.9, 3.0), xytext=(5.4, 3.0), arrowprops=kw)
    ax.annotate("", xy=(8.6, 3.0), xytext=(8.1, 3.0), arrowprops=kw)

    ax.text(5.5, 0.3, "Figure 4.1: Unified End-to-End System Architecture of the Urban Microclimate Modelling Platform.", ha="center", va="center", fontsize=10, fontstyle="italic", color="#334155")

    plt.tight_layout()
    out = OUTPUT_DIR / "system_architecture.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

def generate_data_workflow():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    c_box = "#0284c7"
    boxes = [
        ("Urban Geometry\nBuilding Footprints\nHeights & Locations", 0.5, 2.0, 1.8, 1.5),
        ("7-Channel Rasterizer\nH, Mask, SDF, U_in,\nV_in, X/Lx, Y/Ly", 2.8, 2.0, 1.8, 1.5),
        ("Train-Only Split\nLayout/Family Holdout\nZero Spatial Leakage", 5.1, 2.0, 1.8, 1.5),
        ("Neural Surrogate\nCompact U-Net / CNN\nOutdoor Masking", 7.4, 2.0, 1.8, 1.5),
    ]

    for title, x, y, w, h in boxes:
        ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15", facecolor="#f0f9ff", edgecolor=c_box, linewidth=1.8))
        ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=8.5, fontweight="semibold", color="#0f172a", linespacing=1.4)

    kw = dict(arrowstyle="->", color="#0284c7", lw=2, mutation_scale=15)
    ax.annotate("", xy=(2.8, 2.75), xytext=(2.3, 2.75), arrowprops=kw)
    ax.annotate("", xy=(5.1, 2.75), xytext=(4.6, 2.75), arrowprops=kw)
    ax.annotate("", xy=(7.4, 2.75), xytext=(6.9, 2.75), arrowprops=kw)

    # Bottom parallel branch for Random Forest
    ax.add_patch(patches.FancyBboxPatch((2.8, 0.2), 4.1, 1.1, boxstyle="round,pad=0.15", facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.8))
    ax.text(4.85, 0.75, "Morphometric Feature Extraction (26D Vector)\nRandom Forest Baseline -> Area-Averaged Wind Speed [m/s]", ha="center", va="center", fontsize=8, fontweight="semibold", color="#78350f")

    ax.annotate("", xy=(2.8, 0.75), xytext=(1.4, 2.0), arrowprops=dict(arrowstyle="->", color="#d97706", lw=1.5, connectionstyle="angle,angleA=-90,angleB=180,rad=5"))

    ax.text(5.0, 4.4, "Figure 3.1: Data Preprocessing and Parallel Modeling Workflow (Spatial Surrogate vs Tabular Baseline).", ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#334155")

    plt.tight_layout()
    out = OUTPUT_DIR / "data_workflow.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    generate_system_architecture()
    generate_data_workflow()

