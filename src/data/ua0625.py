"""Real UrbanTALES dataset integration for the UA0625_d00 sample.

This module loads the *real* UrbanTALES LES data shipped with the repository
for a single configuration (``UA0625_d00``) and exposes it through the
``src.data`` package alongside the existing synthetic / canonical fixtures.

Source files (preserved, never modified)
----------------------------------------
* ``UA0625_d00_topo.csv``      -- 2D topology / building-height grid (geometry).
* ``profile-UA0625_d00.csv``   -- 1D horizontally-averaged vertical profile.
* ``UA0625_d00.png``           -- high-resolution geometry rendering (visualisation).

Scientific constraints enforced here
------------------------------------
1. **No fabricated 2D wind field.** The vertical profile is a 1D
   height-dependent, horizontally-averaged quantity. It is *not* a 2D
   pedestrian-level field and is never broadcast into one. The loader
   therefore deliberately exposes ``has_2d_wind_field = False``.
2. **No assumed grid resolution / orientation / spacing.** The topology CSV
   carries no spacing metadata, so ``resolution_m`` and ``domain_size_m`` are
   ``None`` (not guessed). Plan-area density ``lambda_p`` is *computed* from
   the raster (verifiable) and happens to match the filename label.
3. **No CFD / training claims.** The data originates from UrbanTALES LES run
   elsewhere; we did not run the simulation locally (``locally_simulated =
   False``) and no model is trained on this single sample.
4. **Honest units.** The single building height value (16.0) is interpreted as
   metres, consistent with the profile ``z`` range (0-131.5 m) and the
   UrbanTALES convention. The topology file itself contains no unit metadata,
   so this is recorded as an interpretation, not a confirmed file attribute.

The module deliberately does **not** construct a full
:class:`~src.data.contract.UrbanLayoutSample` because that contract requires
physical spacing metadata and a 2D pedestrian-wind-speed target, neither of
which is available for this real sample. Instead the geometry is validated
against the same :class:`~src.data.contract.GeometryData` consistency rules
used by the rest of the pipeline, keeping the integration consistent with the
existing data contract without fabricating the missing pieces.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.data.contract import (
    GeometryData,
    NonFiniteValueError,
    PhysicalConstraintValidationError,
    ShapeMismatchValidationError,
)

__all__ = [
    "UA0625Provenance",
    "UA0625Topology",
    "UA0625VerticalProfile",
    "UA0625Image",
    "UA0625Dataset",
    "parse_ua0625_dataset_id",
    "load_ua0625_topology",
    "load_ua0625_profile",
    "load_ua0625_image",
    "load_ua0625_dataset",
    "default_ua0625_paths",
]

# Physical quantities surfaced from the vertical profile. The header is read
# by name (not by fixed column index) so the loader is robust to column
# re-ordering. These are the columns used for "vertical wind / turbulence
# analysis"; the full header is retained on the dataclass for completeness.
_PROFILE_CORE_COLUMNS: Tuple[str, ...] = ("z", "u", "v", "w", "U", "TKE")

# UrbanTALES reference (Nazarian et al., 2025).
_URBANTALES_CITATION = (
    "UrbanTALES LES (Nazarian et al., 2025; "
    "BAMS DOI: 10.1175/BAMS-D-25-0061.1)"
)


# ---------------------------------------------------------------------------
# Dataset-id parsing
# ---------------------------------------------------------------------------

def parse_ua0625_dataset_id(dataset_id: str) -> Dict[str, object]:
    """Parse a UAxxxx_dyy style dataset id into descriptive labels.

    The naming convention used by the shipped files is ``UA0625_d00``:
    * ``UA``   -- UrbanTALES prefix.
    * ``0625`` -- plan-area-density label (``lambda_p = 0.0625``).
    * ``d00``  -- inflow direction label (``0`` degrees).

    These are *filename conventions*, not file-embedded metadata. The
    ``lambda_p`` label is cross-checked against the value computed from the
    raster by :func:`load_ua0625_topology`. The direction label is **not**
    verifiable from the topology file alone and is returned only as a label.

    Returns a dict with keys ``plan_area_density_label`` (float | None),
    ``wind_direction_label_deg`` (float | None) and ``parsed`` (bool).
    """
    m = re.fullmatch(r"UA(\d+)_d(\d+)", dataset_id.strip())
    if not m:
        return {
            "plan_area_density_label": None,
            "wind_direction_label_deg": None,
            "parsed": False,
        }
    density_digits, dir_digits = m.group(1), m.group(2)
    # 4-digit density code -> decimal fraction (0625 -> 0.0625).
    plan_label: Optional[float]
    if len(density_digits) == 4:
        plan_label = int(density_digits) / 10000.0
    elif len(density_digits) == 3:
        plan_label = int(density_digits) / 1000.0
    else:
        plan_label = None
    wind_label = float(int(dir_digits))
    return {
        "plan_area_density_label": plan_label,
        "wind_direction_label_deg": wind_label,
        "parsed": True,
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UA0625Provenance:
    """Honest provenance and capability flags for the real UrbanTALES sample."""
    dataset_id: str
    source: str = _URBANTALES_CITATION
    license: str = "CC BY-NC 4.0"
    # We loaded published LES output; we did not run the solver locally.
    locally_simulated: bool = False
    # Geometry (2D topology grid) is available.
    has_geometry: bool = True
    # A 1D vertical profile is available.
    has_vertical_profile: bool = True
    # Crucially: NO 2D pedestrian-level wind field is available for this file
    # set, and none is fabricated from the profile.
    has_2d_wind_field: bool = False
    # Filename-derived labels (see parse_ua0625_dataset_id).
    plan_area_density_label: Optional[float] = None
    wind_direction_label_deg: Optional[float] = None
    wind_direction_label_source: str = (
        "filename convention (d00); not verifiable from topology file alone"
    )


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

@dataclass
class UA0625Topology:
    """2D building-topology grid loaded from ``UA0625_d00_topo.csv``.

    The grid is stored as a ``float32`` height map (metres above ground, 0 for
    ground) plus a boolean building mask. Physical grid spacing is **not**
    recorded in the source file, so ``resolution_m`` / ``domain_size_m`` are
    ``None``; only dimensionless / count-based geometry statistics that do not
    depend on absolute spacing are reported by :meth:`geometry_summary`.
    """

    dataset_id: str
    height_map: np.ndarray            # (Ny, Nx) float32, metres (>= 0)
    building_mask: np.ndarray         # (Ny, Nx) bool
    grid_shape: Tuple[int, int]       # (Ny, Nx)
    source_file: str
    # None because the source file carries no spacing metadata.
    resolution_m: Optional[Tuple[float, float, float]] = None
    domain_size_m: Optional[Tuple[float, float, float]] = None
    units_note: str = (
        "Building height value interpreted as metres, consistent with the "
        "vertical profile z-axis range (0-131.5 m) and UrbanTALES conventions; "
        "the topology CSV itself contains no explicit unit metadata."
    )

    # -- derived, lazily computed -------------------------------------------
    @property
    def building_height_values(self) -> np.ndarray:
        """Sorted unique nonzero building-height values present in the grid."""
        nz = self.height_map[self.height_map > 1e-4]
        return np.unique(nz) if nz.size else np.array([], dtype=np.float32)

    @property
    def plan_area_density(self) -> float:
        """Plan-area density lambda_p = built cells / total cells (dimensionless).

        This is computed directly from the raster and is therefore verifiable;
        it is not assumed from the filename.
        """
        total = int(self.height_map.size)
        built = int(np.count_nonzero(self.building_mask))
        return float(built) / float(total) if total else 0.0

    def validate(self) -> None:
        """Validate the topology grid using the pipeline's GeometryData rules."""
        ny, nx = self.grid_shape
        expected = (ny, nx)
        # Reuse the contract's geometry validation (finite, non-negative
        # heights, mask consistency where height > 0).
        geom = GeometryData(height_map=self.height_map, building_mask=self.building_mask)
        geom.validate(expected_shape=expected)

        if self.height_map.shape != expected:
            raise ShapeMismatchValidationError(
                f"height_map shape {self.height_map.shape} != grid_shape {expected}"
            )
        if self.building_mask.shape != expected:
            raise ShapeMismatchValidationError(
                f"building_mask shape {self.building_mask.shape} != grid_shape {expected}"
            )
        # Full two-way mask/height consistency (the contract checks one
        # direction; we additionally require mask==True => height > 0).
        positive = self.height_map > 1e-4
        if np.any(self.building_mask & ~positive):
            raise PhysicalConstraintValidationError(
                "building_mask is True where height_map is ~0 "
                f"({int(np.count_nonzero(self.building_mask & ~positive))} cells)."
            )

    def geometry_summary(self) -> Dict[str, object]:
        """Spacing-independent geometry statistics (no fabricated spacing).

        Returns only quantities computable from the raster without physical
        spacing metadata: grid shape, lambda_p, building count, and building
        height distribution statistics. Spacing-dependent quantities (frontal
        / wall area densities, MacDonald z0, footprint areas in m^2, domain
        extents) are intentionally omitted.
        """
        mask = self.building_mask
        heights = self.height_map[mask]
        from scipy.ndimage import label as ndimage_label  # local import keeps module light
        n_buildings = int(ndimage_label(mask)[1]) if np.any(mask) else 0
        if heights.size:
            hmean = float(np.mean(heights))
            hstd = float(np.std(heights))
            hmin = float(np.min(heights))
            hmax = float(np.max(heights))
            p25 = float(np.percentile(heights, 25))
            p75 = float(np.percentile(heights, 75))
            cv = float(hstd / hmean) if hmean > 1e-9 else 0.0
        else:
            hmean = hstd = hmin = hmax = p25 = p75 = cv = 0.0
        return {
            "grid_shape": list(self.grid_shape),
            "plan_area_density_lambda_p": round(self.plan_area_density, 6),
            "building_count": n_buildings,
            "building_height_values_m": [float(h) for h in self.building_height_values],
            "mean_building_height_m": round(hmean, 4),
            "std_building_height_m": round(hstd, 4),
            "min_building_height_m": round(hmin, 4),
            "max_building_height_m": round(hmax, 4),
            "p25_building_height_m": round(p25, 4),
            "p75_building_height_m": round(p75, 4),
            "height_iqr_m": round(p75 - p25, 4),
            "height_cv": round(cv, 5),
            "resolution_m": None,
            "domain_size_m": None,
            "spacing_available": False,
        }


# ---------------------------------------------------------------------------
# Vertical profile
# ---------------------------------------------------------------------------

@dataclass
class UA0625VerticalProfile:
    """1D horizontally-averaged vertical profile from ``profile-UA0625_d00.csv``.

    The physically meaningful columns used for vertical wind / turbulence
    analysis are exposed as 1D arrays (``z``, ``u``, ``v``, ``w``, ``speed``,
    ``tke``). Missing cells in the source CSV are represented as ``NaN``;
    notably the ground level (z = 0 m) has no defined mean velocity (no-slip),
    so those entries are ``NaN``. The full column header is retained on
    :attr:`column_names`.

    Important: the dataset-reported ``U`` (``speed``) is **not** asserted to
    equal ``sqrt(u^2 + v^2 + w^2)``; the two differ for this dataset, so the
    loader preserves the reported value rather than recomputing it.
    """

    dataset_id: str
    z: np.ndarray           # (N,) float64, metres, strictly increasing
    u: np.ndarray           # (N,) float64, m/s (NaN where undefined)
    v: np.ndarray           # (N,) float64, m/s (NaN where undefined)
    w: np.ndarray           # (N,) float64, m/s (NaN where undefined)
    speed: np.ndarray       # (N,) float64, m/s, dataset-reported 'U'
    tke: np.ndarray         # (N,) float64, m^2/s^2, dataset-reported 'TKE'
    num_levels: int
    column_names: List[str]
    source_file: str
    notes: str = (
        "'U' is the dataset-reported speed magnitude (not recomputed as "
        "sqrt(u^2+v^2+w^2); the two differ for this dataset). 'TKE' may take "
        "negative values per the dataset-specific definition near the domain "
        "top. Ground-level (z=0) velocity/TKE entries are NaN (no-slip)."
    )

    def validate(self) -> None:
        """Validate array shapes and the monotonic height axis.

        We do *not* impose physical sign/relationship constraints (e.g. TKE >= 0
        or speed == sqrt(u^2+v^2+w^2)) because the real dataset violates those
        assumptions; only structural integrity is checked here.
        """
        arrays = (self.z, self.u, self.v, self.w, self.speed, self.tke)
        n = self.num_levels
        for name, arr in zip(("z", "u", "v", "w", "speed", "tke"), arrays):
            if arr.ndim != 1:
                raise ShapeMismatchValidationError(f"{name} must be 1-D, got shape {arr.shape}")
            if arr.shape[0] != n:
                raise ShapeMismatchValidationError(
                    f"{name} length {arr.shape[0]} != num_levels {n}"
                )
        # z must be finite everywhere (it is the independent axis).
        if not np.all(np.isfinite(self.z)):
            raise NonFiniteValueError("profile z-axis contains non-finite values")
        # z must be strictly increasing.
        if self.z.size >= 2 and not np.all(np.diff(self.z) > 0.0):
            raise PhysicalConstraintValidationError(
                "profile z-axis must be strictly increasing"
            )

    @property
    def z_min(self) -> float:
        return float(self.z[0]) if self.z.size else 0.0

    @property
    def z_max(self) -> float:
        return float(self.z[-1]) if self.z.size else 0.0

    def summary(self) -> Dict[str, object]:
        """Return a compact, NaN-aware summary of the profile columns."""
        def stat(a: np.ndarray) -> Dict[str, Optional[float]]:
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                return {"finite_count": 0, "min": None, "max": None, "mean": None}
            return {
                "finite_count": int(finite.size),
                "min": round(float(np.min(finite)), 5),
                "max": round(float(np.max(finite)), 5),
                "mean": round(float(np.mean(finite)), 5),
            }
        return {
            "num_levels": self.num_levels,
            "z_min_m": round(self.z_min, 4),
            "z_max_m": round(self.z_max, 4),
            "u": stat(self.u),
            "v": stat(self.v),
            "w": stat(self.w),
            "speed_U": stat(self.speed),
            "tke": stat(self.tke),
        }


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

@dataclass
class UA0625Image:
    """Geometry rendering loaded from ``UA0625_d00.png`` (visualisation only)."""
    dataset_id: str
    array: np.ndarray       # (H, W, C) uint8
    size: Tuple[int, int]   # (width, height) per PIL convention
    mode: str
    source_file: str


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class UA0625Dataset:
    """Unified container for the three UA0625_d00 artefacts + provenance."""
    dataset_id: str
    topology: UA0625Topology
    profile: UA0625VerticalProfile
    image: Optional[UA0625Image]
    provenance: UA0625Provenance
    limitations: List[str] = field(default_factory=list)

    def validate(self) -> None:
        self.topology.validate()
        self.profile.validate()
        # image has no physical validation beyond load success

    @property
    def has_2d_wind_field(self) -> bool:
        """Explicit guard: no 2D pedestrian wind field is available/fabricated."""
        return self.provenance.has_2d_wind_field


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def default_ua0625_paths(project_root: Optional[Path] = None) -> Dict[str, Path]:
    """Return the default on-disk locations of the three UA0625_d00 files."""
    root = Path(project_root) if project_root is not None else _project_root()
    return {
        "topology": root / "UA0625_d00_topo.csv",
        "profile": root / "profile-UA0625_d00.csv",
        "image": root / "UA0625_d00.png",
    }


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _empty_to_nan(values: List[str]) -> np.ndarray:
    """Convert a list of raw CSV cell strings to a float64 array (empty -> NaN)."""
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        s = v.strip() if isinstance(v, str) else v
        out[i] = np.nan if s == "" else float(s)
    return out


def load_ua0625_topology(
    csv_path: Path | str,
    dataset_id: str = "UA0625_d00",
) -> UA0625Topology:
    """Load the 2D topology grid from ``UA0625_d00_topo.csv``.

    The file is a header-less numeric grid (rows = y, columns = x) of building
    heights above ground. Ground is 0; building cells carry the building
    height value (metres, interpreted -- see :class:`UA0625Topology`).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Topology CSV not found: {path}")
    raw = np.loadtxt(path, delimiter=",")
    if raw.ndim != 2:
        raise ShapeMismatchValidationError(
            f"Expected a 2-D topology grid, got shape {raw.shape}"
        )
    height_map = raw.astype(np.float32)
    building_mask = height_map > 1e-4
    ny, nx = height_map.shape

    topo = UA0625Topology(
        dataset_id=dataset_id,
        height_map=height_map,
        building_mask=building_mask,
        grid_shape=(ny, nx),
        source_file=str(path),
    )
    topo.validate()

    # Cross-check the filename lambda_p label against the computed value.
    labels = parse_ua0625_dataset_id(dataset_id)
    plan_label = labels["plan_area_density_label"]
    if plan_label is not None:
        computed = topo.plan_area_density
        if abs(computed - float(plan_label)) > 1e-6:
            raise PhysicalConstraintValidationError(
                f"Filename lambda_p label {plan_label} does not match the value "
                f"computed from the raster ({computed:.6f})."
            )
    return topo


def load_ua0625_profile(
    csv_path: Path | str,
    dataset_id: str = "UA0625_d00",
) -> UA0625VerticalProfile:
    """Load the 1D vertical profile from ``profile-UA0625_d00.csv``.

    Columns are read by header name. The core physical columns
    (``z, u, v, w, U, TKE``) are exposed as arrays; missing cells become NaN.
    The dataset-reported ``U`` is preserved verbatim (not recomputed).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Profile CSV not found: {path}")

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if not header or not rows:
        raise ShapeMismatchValidationError(f"Profile CSV {path} is empty or has no data rows.")

    col_index = {name.strip(): i for i, name in enumerate(header)}

    def column(name: str) -> np.ndarray:
        if name not in col_index:
            raise KeyError(f"Column '{name}' not found in profile header: {header}")
        return _empty_to_nan([row[col_index[name]] for row in rows])

    z = column("z")
    u = column("u")
    v = column("v")
    w = column("w")
    speed = column("U")
    tke = column("TKE")
    n = z.shape[0]

    profile = UA0625VerticalProfile(
        dataset_id=dataset_id,
        z=z,
        u=u,
        v=v,
        w=w,
        speed=speed,
        tke=tke,
        num_levels=n,
        column_names=[h.strip() for h in header],
        source_file=str(path),
    )
    profile.validate()
    return profile


def load_ua0625_image(
    png_path: Path | str,
    dataset_id: str = "UA0625_d00",
) -> UA0625Image:
    """Load the geometry rendering PNG (visualisation only)."""
    from PIL import Image  # imported lazily so the data package imports without PIL

    path = Path(png_path)
    if not path.exists():
        raise FileNotFoundError(f"Image PNG not found: {path}")
    img = Image.open(path)
    arr = np.array(img)
    return UA0625Image(
        dataset_id=dataset_id,
        array=arr,
        size=img.size,        # (width, height)
        mode=img.mode,
        source_file=str(path),
    )


def load_ua0625_dataset(
    topo_csv: Optional[Path | str] = None,
    profile_csv: Optional[Path | str] = None,
    image_png: Optional[Path | str] = None,
    dataset_id: str = "UA0625_d00",
    project_root: Optional[Path] = None,
    load_image: bool = True,
) -> UA0625Dataset:
    """Load the full UA0625_d00 dataset (topology + profile + optional image).

    Any path left ``None`` falls back to the default repo location. If the
    image file is absent and ``load_image`` is True, the image field is set to
    ``None`` rather than raising, so the dataset is still usable for analysis.
    """
    defaults = default_ua0625_paths(project_root)
    topo_csv = Path(topo_csv) if topo_csv is not None else defaults["topology"]
    profile_csv = Path(profile_csv) if profile_csv is not None else defaults["profile"]
    image_png = Path(image_png) if image_png is not None else defaults["image"]

    topology = load_ua0625_topology(topo_csv, dataset_id=dataset_id)
    profile = load_ua0625_profile(profile_csv, dataset_id=dataset_id)

    image: Optional[UA0625Image] = None
    if load_image and image_png.exists():
        image = load_ua0625_image(image_png, dataset_id=dataset_id)

    labels = parse_ua0625_dataset_id(dataset_id)
    provenance = UA0625Provenance(
        dataset_id=dataset_id,
        plan_area_density_label=labels["plan_area_density_label"],
        wind_direction_label_deg=labels["wind_direction_label_deg"],
    )

    limitations = [
        "No 2D pedestrian-level wind field is available; the vertical profile "
        "is a 1D horizontally-averaged quantity and is never broadcast into a 2D field.",
        "Grid resolution / physical spacing is not recorded in the topology CSV "
        "and is therefore unknown (resolution_m = domain_size_m = None).",
        "Wind-direction label derives from the filename convention only and is "
        "not verifiable from the topology file alone.",
        "Building height is interpreted as metres (UrbanTALES convention); the "
        "topology CSV carries no explicit unit metadata.",
        "The dataset-reported 'U' is not equal to sqrt(u^2+v^2+w^2) for this "
        "sample and is preserved verbatim; 'TKE' may be negative near the domain top.",
        "No CFD was run locally and no model is trained on this single sample.",
    ]

    dataset = UA0625Dataset(
        dataset_id=dataset_id,
        topology=topology,
        profile=profile,
        image=image,
        provenance=provenance,
        limitations=limitations,
    )
    dataset.validate()
    return dataset
