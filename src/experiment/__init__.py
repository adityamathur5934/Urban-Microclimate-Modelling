"""Reproducible CFD-to-neural-surrogate experiment orchestration.

This package wires the existing data, model, and evaluation components into a
single reproducible experiment. The reference fields used for training are
produced by a documented *analytical* aerodynamic model (Castro & Robins 1977
formulation) and are **explicitly not** CFD solver output — see
``reference.py`` and ``docs/cfd_surrogate_experiment.md`` for the honest
provenance story. Real OpenFOAM cases can be generated with
``cfd/scripts/generate_cfd_cases.py`` and consumed through the same pipeline
via ``cfd/scripts/foam_to_contract.py`` once a solver is available.
"""

from src.experiment.reference import (
    build_analytical_reference_sample,
    generate_reference_dataset,
    analytical_pedestrian_wind_field,
    REFERENCE_PROVENANCE,
)
from src.experiment.dataset import (
    ExperimentDataset,
    fit_field_normalizers,
)
from src.experiment.uncertainty import ensemble_predict, predictive_disagreement
from src.experiment.runner import (
    ExperimentConfig,
    ExperimentResult,
    run_experiment,
)

__all__ = [
    "build_analytical_reference_sample",
    "generate_reference_dataset",
    "analytical_pedestrian_wind_field",
    "REFERENCE_PROVENANCE",
    "ExperimentDataset",
    "fit_field_normalizers",
    "ensemble_predict",
    "predictive_disagreement",
    "ExperimentConfig",
    "ExperimentResult",
    "run_experiment",
]
