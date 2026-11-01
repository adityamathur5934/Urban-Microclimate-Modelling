#!/usr/bin/env python3
"""Exploratory Data Analysis (EDA) for the real UrbanTALES UA0625_d00 dataset.

Generates a focused set of visualisations into ``outputs/ua0625/``:

1. ``ua0625_01_topology_height.png``  -- 2D building-height topology grid.
2. ``ua0625_02_building_mask.png``     -- binary building-footprint mask.
3. ``ua0625_03_vertical_wind_profile.png`` -- u, v, w, U(speed) vs height.
4. ``ua0625_04_tke_profile.png``       -- turbulent kinetic energy vs height.
5. ``ua0625_05_geometry_image.png``    -- the geometry rendering (PNG) used
   for geometry visualisation (downscaled for display; original preserved).

Scientific honesty notes
-------------------------
* No 2D wind field is fabricated from the 1D vertical profile -- the wind and
  TKE plots are strictly the horizontally-averaged vertical profile.
* The topology heatmap shows cell indices (not metres) because the source
  CSV carries no physical spacing metadata; this is annotated on the figure.
* The reported speed ``U`` is plotted verbatim (it is NOT recomputed as
  sqrt(u^2+v^2+w^2) for this dataset).

Usage:
    python scripts/ua0625_eda.py
    python scripts/ua0625_eda.py --output-dir outputs/ua0625
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from src.data.ua0625 import load_ua0625_dataset, default_ua0625_paths


def _annotate_spacing_note(fig) -> None:
    """Add a footer noting that physical spacing is not available."""
    fig.text(
        0.5,
        0.01,
        "Axes shown in grid-cell indices: the topology CSV carries no physical "
        "spacing metadata (resolution_m = None).",
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="0.35",
    )


def plot_topology_height(topo, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    im = ax.imshow(
        topo.height_map,
        origin="lower",
        cmap="viridis",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Building height [m] (interpreted)")
    ax.set_title(
        f"{topo.dataset_id} -- topology grid "
        f"({topo.grid_shape[1]} x {topo.grid_shape[0]} cells, "
        f"lambda_p = {topo.plan_area_density:.4f}, "
        f"{int(np.count_nonzero(topo.building_mask))} building cells)"
    )
    ax.set_xlabel("x cell index")
    ax.set_ylabel("y cell index")
    _annotate_spacing_note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_building_mask(topo, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    cmap = ListedColormap(["#f7f7f7", "#2b2b2b"])  # fluid=light, building=dark
    ax.imshow(
        topo.building_mask.astype(np.uint8),
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"{topo.dataset_id} -- building footprint mask "
        f"(white = fluid, black = building)"
    )
    ax.set_xlabel("x cell index")
    ax.set_ylabel("y cell index")
    _annotate_spacing_note(fig)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_vertical_wind_profile(profile, out_path: Path) -> None:
    z = profile.z
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    # Plot horizontally-averaged velocity components vs height.
    ax.plot(profile.u, z, label="u (streamwise)", lw=1.8)
    ax.plot(profile.v, z, label="v (spanwise)", lw=1.6)
    ax.plot(profile.w, z, label="w (vertical)", lw=1.6)
    ax.plot(profile.speed, z, label="U (dataset-reported speed)", lw=2.0, ls="--")
    ax.set_xlabel("Velocity [m/s]")
    ax.set_ylabel("Height z [m]")
    ax.set_title(
        f"{profile.dataset_id} -- vertical wind profile (horizontally averaged)\n"
        f"{profile.num_levels} levels, z = {profile.z_min:.2f}-{profile.z_max:.1f} m"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    # Mark no-slip ground (NaN at z=0)
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    fig.text(
        0.5,
        0.01,
        "U is plotted verbatim (not recomputed as sqrt(u^2+v^2+w^2)); "
        "ground-level (z=0) entries are NaN (no-slip).",
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_tke_profile(profile, out_path: Path) -> None:
    z = profile.z
    tke = profile.tke
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    ax.plot(tke, z, color="#b35806", lw=1.8)
    ax.set_xlabel("TKE [m^2/s^2] (dataset-reported)")
    ax.set_ylabel("Height z [m]")
    ax.set_title(f"{profile.dataset_id} -- turbulent kinetic energy profile")
    ax.grid(alpha=0.3)
    ax.axvline(0.0, color="0.6", lw=0.8, ls=":")
    fig.text(
        0.5,
        0.01,
        "Dataset-reported 'TKE' may take negative values near the domain top "
        "(dataset-specific definition); values are plotted as-is.",
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="0.35",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_geometry_image(image, out_path: Path) -> None:
    """Render the geometry PNG (downscaled) for visualisation; original preserved."""
    from PIL import Image

    img = Image.open(image.source_file)
    # Downscale for a lightweight figure (original file is untouched).
    max_w = 2400
    if img.size[0] > max_w:
        ratio = max_w / img.size[0]
        img = img.resize((max_w, int(round(img.size[1] * ratio))))
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.imshow(np.array(img))
    ax.set_title(
        f"{image.dataset_id} -- geometry rendering "
        f"(source: {Path(image.source_file).name}, "
        f"original {image.size[0]}x{image.size[1]} px {image.mode})"
    )
    ax.set_xlabel("x pixel")
    ax.set_ylabel("y pixel")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate EDA visualisations for the real UrbanTALES UA0625_d00 dataset."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "ua0625"),
        help="Directory to write figures (default: outputs/ua0625).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = default_ua0625_paths(PROJECT_ROOT)
    print("Loading UA0625_d00 dataset ...")
    print(f"  topology : {paths['topology']}")
    print(f"  profile  : {paths['profile']}")
    print(f"  image    : {paths['image']}")

    ds = load_ua0625_dataset(project_root=PROJECT_ROOT)
    print(
        f"  -> topology {ds.topology.grid_shape[1]}x{ds.topology.grid_shape[0]}, "
        f"lambda_p={ds.topology.plan_area_density:.4f}, "
        f"buildings={ds.topology.geometry_summary()['building_count']}"
    )
    print(
        f"  -> profile {ds.profile.num_levels} levels, "
        f"z={ds.profile.z_min:.2f}-{ds.profile.z_max:.1f} m"
    )

    written = []

    f1 = out_dir / "ua0625_01_topology_height.png"
    plot_topology_height(ds.topology, f1)
    written.append(f1)

    f2 = out_dir / "ua0625_02_building_mask.png"
    plot_building_mask(ds.topology, f2)
    written.append(f2)

    f3 = out_dir / "ua0625_03_vertical_wind_profile.png"
    plot_vertical_wind_profile(ds.profile, f3)
    written.append(f3)

    f4 = out_dir / "ua0625_04_tke_profile.png"
    plot_tke_profile(ds.profile, f4)
    written.append(f4)

    if ds.image is not None:
        f5 = out_dir / "ua0625_05_geometry_image.png"
        plot_geometry_image(ds.image, f5)
        written.append(f5)

    print("\nEDA figures written:")
    for f in written:
        print(f"  {f}")
    print("\nProvenance:", ds.provenance.source)
    print("Limitations:")
    for lim in ds.limitations:
        print(f"  - {lim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
