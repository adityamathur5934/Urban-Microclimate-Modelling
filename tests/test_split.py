"""Tests for data splitting, reproducibility, metadata export, and prevention of layout leakage."""

from pathlib import Path
import pytest

from src.data.split import LayoutSplitter, DatasetSplit
from src.data.contract import LayoutFamily
from tests.fixtures.synthetic_data import (
    create_synthetic_sample,
    create_synthetic_split_dataset_fixture,
)


def test_split_by_ratio_no_overlap():
    """Verify ratio splitter produces disjoint sets across train, val, and test."""
    layout_ids = [f"layout_{i:03d}" for i in range(20)]
    split = LayoutSplitter.split_by_ratio(
        layout_ids, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=123
    )

    split.validate()
    split.assert_no_overlap()
    train_set = set(split.train_layout_ids)
    val_set = set(split.val_layout_ids)
    test_set = set(split.test_layout_ids)

    # Check zero overlap
    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0

    # Check total coverage
    assert len(train_set | val_set | test_set) == 20
    assert split.total_layouts == 20


def test_split_reproducibility_with_seed():
    """Verify identical seeds produce identical splits and different seeds yield different splits."""
    layout_ids = [f"layout_{i:03d}" for i in range(30)]

    split_seed42_a = LayoutSplitter.split_by_ratio(layout_ids, seed=42)
    split_seed42_b = LayoutSplitter.split_by_ratio(layout_ids, seed=42)
    split_seed99 = LayoutSplitter.split_by_ratio(layout_ids, seed=99)

    # Exactly identical partitions for identical seeds
    assert split_seed42_a.train_layout_ids == split_seed42_b.train_layout_ids
    assert split_seed42_a.val_layout_ids == split_seed42_b.val_layout_ids
    assert split_seed42_a.test_layout_ids == split_seed42_b.test_layout_ids

    # Different seeds shuffle differently
    assert split_seed42_a.train_layout_ids != split_seed99.train_layout_ids


def test_split_by_group_holds_out_related_layouts():
    """Verify that multi-run / related layout variations strictly remain in the same split."""
    fixture = create_synthetic_split_dataset_fixture()
    layout_to_group = fixture["layout_to_group"]

    split = LayoutSplitter.split_by_group(
        layout_to_group=layout_to_group,
        train_ratio=0.50,
        val_ratio=0.25,
        test_ratio=0.25,
        seed=42,
    )

    split.validate()
    split.assert_no_overlap(check_families=True)

    # Check that all layouts in each group belong to one single partition
    train_set = set(split.train_layout_ids)
    val_set = set(split.val_layout_ids)
    test_set = set(split.test_layout_ids)

    for group_id in set(layout_to_group.values()):
        group_layouts = {lid for lid, gid in layout_to_group.items() if gid == group_id}
        # Group must be entirely in train, entirely in val, or entirely in test
        in_train = group_layouts.issubset(train_set)
        in_val = group_layouts.issubset(val_set)
        in_test = group_layouts.issubset(test_set)
        assert in_train or in_val or in_test, f"Group {group_id} was split across partitions!"


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
    split.assert_no_overlap(check_families=True)

    # All realistic neighborhoods must be in test only
    expected_test_ids = {f"real_{i}" for i in range(4)}
    assert set(split.test_layout_ids) == expected_test_ids

    # Train and val must contain only idealized families
    for lid in split.train_layout_ids + split.val_layout_ids:
        assert not lid.startswith("real_")


def test_split_metadata_json_serialization_roundtrip(tmp_path: Path):
    """Verify split metadata and partitions save to JSON and load with full fidelity."""
    layout_ids = [f"layout_{i:03d}" for i in range(25)]
    split = LayoutSplitter.split_by_ratio(
        layout_ids, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=777
    )

    json_path = tmp_path / "dataset_split.json"
    saved_path = split.save_json(json_path)
    assert saved_path.exists()

    loaded = DatasetSplit.load_json(saved_path)
    assert loaded.train_layout_ids == split.train_layout_ids
    assert loaded.val_layout_ids == split.val_layout_ids
    assert loaded.test_layout_ids == split.test_layout_ids
    assert loaded.metadata["seed"] == 777
    assert loaded.metadata["split_strategy"] == "ratio"

    summary = split.summary()
    assert summary["total_layouts"] == 25
    assert summary["train_count"] == len(split.train_layout_ids)


def test_leakage_detection_raises_error():
    """DatasetSplit validation must throw ValueError if an ID exists in multiple splits."""
    with pytest.raises(ValueError, match="Data leakage detected"):
        DatasetSplit(
            train_layout_ids=["layout_01", "layout_02"],
            val_layout_ids=["layout_02", "layout_03"],
            test_layout_ids=["layout_04"],
        )


def test_group_leakage_detection_raises_error():
    """assert_no_overlap must raise ValueError if a group is split across partitions."""
    mapping = {
        "layout_01": "group_A",
        "layout_02": "group_A",  # Leaked into val!
        "layout_03": "group_B",
    }
    with pytest.raises(ValueError, match="Group leakage detected"):
        DatasetSplit(
            train_layout_ids=["layout_01"],
            val_layout_ids=["layout_02"],
            test_layout_ids=["layout_03"],
            metadata={"split_strategy": "group_holdout"},
            layout_to_family=mapping,
        )


def test_split_by_ratio_invalid_ratios_raise_error():
    """Invalid ratio sums must raise ValueError."""
    with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
        LayoutSplitter.split_by_ratio(["l1", "l2"], train_ratio=0.5, val_ratio=0.2, test_ratio=0.1)
