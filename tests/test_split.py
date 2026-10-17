"""Tests for data splitting and prevention of layout leakage."""

import pytest

from src.data.split import LayoutSplitter, DatasetSplit
from src.data.contract import LayoutFamily
from tests.fixtures.synthetic_data import create_synthetic_sample


def test_split_by_ratio_no_overlap():
    """Verify ratio splitter produces disjoint sets across train, val, and test."""
    layout_ids = [f"layout_{i:03d}" for i in range(20)]
    split = LayoutSplitter.split_by_ratio(
        layout_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=123
    )

    split.validate()
    train_set = set(split.train_layout_ids)
    val_set = set(split.val_layout_ids)
    test_set = set(split.test_layout_ids)

    # Check zero overlap
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Check total coverage
    assert len(train_set | val_set | test_set) == 20


def test_split_by_family_holds_out_entire_group():
    """Verify that an entire layout family can be held out exclusively for testing."""
    samples = [
        create_synthetic_sample(f"ideal_a_{i}", layout_family=LayoutFamily.IDEALIZED_ALIGNED)
        for i in range(5)
    ] + [
        create_synthetic_sample(f"ideal_s_{i}", layout_family=LayoutFamily.IDEALIZED_STAGGERED)
        for i in range(5)
    ] + [
        create_synthetic_sample(f"real_{i}", layout_family=LayoutFamily.REALISTIC_NEIGHBORHOOD)
        for i in range(4)
    ]

    split = LayoutSplitter.split_by_family(
        samples=samples,
        test_families=[LayoutFamily.REALISTIC_NEIGHBORHOOD],
        val_ratio=0.2,
    )

    split.validate()

    # All realistic neighborhoods must be in test only
    expected_test_ids = {f"real_{i}" for i in range(4)}
    assert set(split.test_layout_ids) == expected_test_ids

    # Train and val must contain only idealized families
    for lid in split.train_layout_ids + split.val_layout_ids:
        assert not lid.startswith("real_")


def test_leakage_detection_raises_error():
    """DatasetSplit validation must throw ValueError if an ID exists in multiple splits."""
    with pytest.raises(ValueError, match="Data leakage detected"):
        DatasetSplit(
            train_layout_ids=["layout_01", "layout_02"],
            val_layout_ids=["layout_02", "layout_03"],
            test_layout_ids=["layout_04"],
        )
