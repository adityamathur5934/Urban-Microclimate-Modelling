"""Dataset partitioning logic ensuring layout-level and family-level holdout."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import random

from src.data.contract import UrbanLayoutSample, LayoutFamily


@dataclass
class DatasetSplit:
    """Holds partitioned layout identifiers for training, validation, and testing."""
    train_layout_ids: List[str]
    val_layout_ids: List[str]
    test_layout_ids: List[str]

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Strict verification that no layout ID leaks across splits."""
        train_set = set(self.train_layout_ids)
        val_set = set(self.val_layout_ids)
        test_set = set(self.test_layout_ids)

        train_val_overlap = train_set.intersection(val_set)
        if train_val_overlap:
            raise ValueError(f"Data leakage detected! Train and Val share layouts: {train_val_overlap}")

        train_test_overlap = train_set.intersection(test_set)
        if train_test_overlap:
            raise ValueError(f"Data leakage detected! Train and Test share layouts: {train_test_overlap}")

        val_test_overlap = val_set.intersection(test_set)
        if val_test_overlap:
            raise ValueError(f"Data leakage detected! Val and Test share layouts: {val_test_overlap}")

    def to_dict(self) -> Dict[str, List[str]]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> DatasetSplit:
        return cls(
            train_layout_ids=data["train_layout_ids"],
            val_layout_ids=data["val_layout_ids"],
            test_layout_ids=data["test_layout_ids"],
        )


class LayoutSplitter:
    """Partitions urban layout datasets cleanly without spatial leakage."""

    @staticmethod
    def split_by_ratio(
        layout_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> DatasetSplit:
        """Splits unique layout IDs by specified ratios using a fixed seed."""
        total = train_ratio + val_ratio + test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        unique_ids = sorted(list(set(layout_ids)))
        rng = random.Random(seed)
        shuffled = unique_ids.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))

        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train : n_train + n_val]
        test_ids = shuffled[n_train + n_val :]

        # Edge case: if test is empty due to small n, ensure at least 1 if n >= 3
        if n >= 3:
            if not test_ids and len(train_ids) > 1:
                test_ids.append(train_ids.pop())
            if not val_ids and len(train_ids) > 1:
                val_ids.append(train_ids.pop())

        return DatasetSplit(
            train_layout_ids=train_ids,
            val_layout_ids=val_ids,
            test_layout_ids=test_ids,
        )

    @staticmethod
    def split_by_family(
        samples: List[UrbanLayoutSample],
        test_families: List[LayoutFamily],
        val_ratio: float = 0.2,
        seed: int = 42,
    ) -> DatasetSplit:
        """Holds out one or more layout families completely for generalization testing."""
        test_families_set = set(test_families)
        train_pool_ids: List[str] = []
        test_ids: List[str] = []

        for sample in samples:
            if sample.layout_family in test_families_set:
                test_ids.append(sample.layout_id)
            else:
                train_pool_ids.append(sample.layout_id)

        train_pool_ids = sorted(list(set(train_pool_ids)))
        test_ids = sorted(list(set(test_ids)))

        rng = random.Random(seed)
        shuffled_train_pool = train_pool_ids.copy()
        rng.shuffle(shuffled_train_pool)

        n_val = int(round(len(shuffled_train_pool) * val_ratio))
        val_ids = shuffled_train_pool[:n_val]
        train_ids = shuffled_train_pool[n_val:]

        return DatasetSplit(
            train_layout_ids=train_ids,
            val_layout_ids=val_ids,
            test_layout_ids=test_ids,
        )
