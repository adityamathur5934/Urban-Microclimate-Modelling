"""Dataset partitioning logic ensuring layout-level and family-level holdout."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import random

from src.data.contract import UrbanLayoutSample, LayoutFamily


@dataclass
class DatasetSplit:
    """Holds partitioned layout identifiers for training, validation, and testing.

    Guarantees strict isolation of layouts and/or layout families to prevent
    spatial autocorrelation leakage across splits.
    """
    train_layout_ids: List[str]
    val_layout_ids: List[str]
    test_layout_ids: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    layout_to_family: Optional[Dict[str, str]] = None

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Strict verification that no layout ID leaks across splits."""
        self.assert_no_overlap(check_families=True)

    def assert_no_overlap(self, check_families: bool = True) -> None:
        """Asserts zero overlap between train, val, and test splits.

        Checks:
          1. Zero layout ID overlap between all split pairs.
          2. Zero layout family/group overlap if family mapping or metadata is present.

        Raises:
            ValueError: If any identifier is shared across splits.
        """
        train_set = set(self.train_layout_ids)
        val_set = set(self.val_layout_ids)
        test_set = set(self.test_layout_ids)

        train_val_overlap = train_set.intersection(val_set)
        if train_val_overlap:
            raise ValueError(f"Data leakage detected! Train and Val share layouts: {sorted(train_val_overlap)}")

        train_test_overlap = train_set.intersection(test_set)
        if train_test_overlap:
            raise ValueError(f"Data leakage detected! Train and Test share layouts: {sorted(train_test_overlap)}")

        val_test_overlap = val_set.intersection(test_set)
        if val_test_overlap:
            raise ValueError(f"Data leakage detected! Val and Test share layouts: {sorted(val_test_overlap)}")

        # Family / group isolation check when mapping is available
        if check_families and self.layout_to_family is not None:
            train_families = {self.layout_to_family[lid] for lid in train_set if lid in self.layout_to_family}
            val_families = {self.layout_to_family[lid] for lid in val_set if lid in self.layout_to_family}
            test_families = {self.layout_to_family[lid] for lid in test_set if lid in self.layout_to_family}

            # Check if this split was created under family/group holdout strategy
            strategy = self.metadata.get("split_strategy", "")
            if strategy == "family_holdout":
                test_fams = set(self.metadata.get("test_families", []))
                if test_fams:
                    t_leak = train_families.intersection(test_fams)
                    if t_leak:
                        raise ValueError(f"Family leakage detected! Train shares test families: {sorted(t_leak)}")
                    v_leak = val_families.intersection(test_fams)
                    if v_leak:
                        raise ValueError(f"Family leakage detected! Val shares test families: {sorted(v_leak)}")
                val_fams = set(self.metadata.get("val_families", []))
                if val_fams:
                    tv_leak = train_families.intersection(val_fams)
                    if tv_leak:
                        raise ValueError(f"Family leakage detected! Train shares val families: {sorted(tv_leak)}")
            elif strategy == "group_holdout":
                tf_vf = train_families.intersection(val_families)
                if tf_vf:
                    raise ValueError(f"Group leakage detected! Train and Val share groups: {sorted(tf_vf)}")
                tf_testf = train_families.intersection(test_families)
                if tf_testf:
                    raise ValueError(f"Group leakage detected! Train and Test share groups: {sorted(tf_testf)}")
                vf_testf = val_families.intersection(test_families)
                if vf_testf:
                    raise ValueError(f"Group leakage detected! Val and Test share groups: {sorted(vf_testf)}")

    @property
    def total_layouts(self) -> int:
        return len(self.train_layout_ids) + len(self.val_layout_ids) + len(self.test_layout_ids)

    def get_split_for_layout(self, layout_id: str) -> str:
        """Returns the partition ('train', 'val', or 'test') containing layout_id."""
        if layout_id in set(self.train_layout_ids):
            return "train"
        if layout_id in set(self.val_layout_ids):
            return "val"
        if layout_id in set(self.test_layout_ids):
            return "test"
        raise KeyError(f"Layout ID '{layout_id}' is not present in any split partition.")

    def summary(self) -> Dict[str, Any]:
        """Returns summary statistics for the dataset split."""
        n_tot = self.total_layouts
        return {
            "total_layouts": n_tot,
            "train_count": len(self.train_layout_ids),
            "val_count": len(self.val_layout_ids),
            "test_count": len(self.test_layout_ids),
            "train_percent": round(len(self.train_layout_ids) / n_tot * 100.0, 2) if n_tot > 0 else 0.0,
            "val_percent": round(len(self.val_layout_ids) / n_tot * 100.0, 2) if n_tot > 0 else 0.0,
            "test_percent": round(len(self.test_layout_ids) / n_tot * 100.0, 2) if n_tot > 0 else 0.0,
            "metadata": self.metadata,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes split to dictionary."""
        data: Dict[str, Any] = {
            "train_layout_ids": list(self.train_layout_ids),
            "val_layout_ids": list(self.val_layout_ids),
            "test_layout_ids": list(self.test_layout_ids),
            "metadata": dict(self.metadata),
        }
        if self.layout_to_family is not None:
            data["layout_to_family"] = dict(self.layout_to_family)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DatasetSplit:
        """Instantiates a DatasetSplit from a dictionary."""
        return cls(
            train_layout_ids=data["train_layout_ids"],
            val_layout_ids=data["val_layout_ids"],
            test_layout_ids=data["test_layout_ids"],
            metadata=data.get("metadata", {}),
            layout_to_family=data.get("layout_to_family"),
        )

    def save_json(self, output_path: Union[str, Path]) -> Path:
        """Exports split assignments and metadata to a validated JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    @classmethod
    def load_json(cls, input_path: Union[str, Path]) -> DatasetSplit:
        """Loads and validates a DatasetSplit from a JSON file."""
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Split metadata file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        split = cls.from_dict(data)
        split.validate()
        return split


class LayoutSplitter:
    """Partitions urban layout datasets cleanly without spatial leakage."""

    @staticmethod
    def split_by_ratio(
        layout_ids: List[str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        group_by: Optional[Dict[str, str]] = None,
    ) -> DatasetSplit:
        """Splits unique layout IDs by specified ratios using a fixed seed.

        If group_by mapping (layout_id -> group_id) is provided, entire related
        layouts sharing the same group_id are kept in the same partition.
        """
        total = train_ratio + val_ratio + test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        unique_ids = sorted(list(set(layout_ids)))
        if not unique_ids:
            raise ValueError("Cannot split empty layout_ids list.")

        # If grouping is specified, partition groups rather than individual layouts
        if group_by is not None:
            return LayoutSplitter.split_by_group(
                layout_to_group={lid: group_by.get(lid, lid) for lid in unique_ids},
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                seed=seed,
            )

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

        metadata = {
            "split_strategy": "ratio",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return DatasetSplit(
            train_layout_ids=sorted(train_ids),
            val_layout_ids=sorted(val_ids),
            test_layout_ids=sorted(test_ids),
            metadata=metadata,
        )

    @staticmethod
    def split_by_group(
        layout_to_group: Dict[str, str],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> DatasetSplit:
        """Partitions related layouts by group identifier.

        All layouts sharing a common group ID (e.g. base geometry, neighborhood ID,
        or parent family) are guaranteed to remain strictly in the same split.
        """
        total = train_ratio + val_ratio + test_ratio
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")

        if not layout_to_group:
            raise ValueError("layout_to_group mapping cannot be empty.")

        # Group layouts by group key
        group_to_layouts: Dict[str, List[str]] = {}
        for lid, gid in sorted(layout_to_group.items()):
            group_to_layouts.setdefault(str(gid), []).append(lid)

        unique_groups = sorted(list(group_to_layouts.keys()))
        rng = random.Random(seed)
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)

        n_groups = len(shuffled_groups)
        n_train = int(round(n_groups * train_ratio))
        n_val = int(round(n_groups * val_ratio))

        train_groups = set(shuffled_groups[:n_train])
        val_groups = set(shuffled_groups[n_train : n_train + n_val])
        test_groups = set(shuffled_groups[n_train + n_val :])

        # Edge case: if test is empty due to small n_groups >= 3
        if n_groups >= 3:
            if not test_groups and len(train_groups) > 1:
                popped = shuffled_groups[0]
                train_groups.remove(popped)
                test_groups.add(popped)
            if not val_groups and len(train_groups) > 1:
                popped = shuffled_groups[1]
                train_groups.remove(popped)
                val_groups.add(popped)

        train_ids: List[str] = []
        for g in sorted(train_groups):
            train_ids.extend(group_to_layouts[g])

        val_ids: List[str] = []
        for g in sorted(val_groups):
            val_ids.extend(group_to_layouts[g])

        test_ids: List[str] = []
        for g in sorted(test_groups):
            test_ids.extend(group_to_layouts[g])

        metadata = {
            "split_strategy": "group_holdout",
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "num_groups": n_groups,
            "train_groups": sorted(list(train_groups)),
            "val_groups": sorted(list(val_groups)),
            "test_groups": sorted(list(test_groups)),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        return DatasetSplit(
            train_layout_ids=sorted(train_ids),
            val_layout_ids=sorted(val_ids),
            test_layout_ids=sorted(test_ids),
            metadata=metadata,
            layout_to_family=layout_to_group,
        )

    @staticmethod
    def split_by_family(
        samples: Optional[List[UrbanLayoutSample]] = None,
        layout_to_family: Optional[Dict[str, Union[LayoutFamily, str]]] = None,
        test_families: Optional[List[Union[LayoutFamily, str]]] = None,
        val_families: Optional[List[Union[LayoutFamily, str]]] = None,
        val_ratio: float = 0.20,
        seed: int = 42,
    ) -> DatasetSplit:
        """Holds out entire layout families completely for generalization testing.

        Can be called either with a list of UrbanLayoutSamples or a layout_to_family mapping.
        Entire morphological families (e.g. realistic_neighborhood) are held out exclusively,
        ensuring out-of-distribution evaluation without data leakage.
        """
        # Build normalized layout_to_family mapping
        mapping: Dict[str, str] = {}
        if samples is not None:
            for s in samples:
                fam_str = s.layout_family.value if isinstance(s.layout_family, LayoutFamily) else str(s.layout_family)
                mapping[s.layout_id] = fam_str
        elif layout_to_family is not None:
            for lid, fam in layout_to_family.items():
                fam_str = fam.value if isinstance(fam, LayoutFamily) else str(fam)
                mapping[lid] = fam_str
        else:
            raise ValueError("Either 'samples' or 'layout_to_family' must be provided to split_by_family.")

        if not mapping:
            raise ValueError("Cannot split empty dataset.")

        # If test_families is specified, hold out those families in test
        if test_families is not None:
            test_fam_set = {f.value if isinstance(f, LayoutFamily) else str(f) for f in test_families}
            val_fam_set = {f.value if isinstance(f, LayoutFamily) else str(f) for f in val_families} if val_families is not None else set()

            train_pool_ids: List[str] = []
            val_ids: List[str] = []
            test_ids: List[str] = []

            for lid, fam in sorted(mapping.items()):
                if fam in test_fam_set:
                    test_ids.append(lid)
                elif fam in val_fam_set:
                    val_ids.append(lid)
                else:
                    train_pool_ids.append(lid)

            # If val_families not explicitly specified, partition remaining train pool by val_ratio
            if val_families is None:
                train_pool_ids = sorted(list(set(train_pool_ids)))
                rng = random.Random(seed)
                shuffled_train_pool = train_pool_ids.copy()
                rng.shuffle(shuffled_train_pool)

                n_val = int(round(len(shuffled_train_pool) * val_ratio))
                val_ids = shuffled_train_pool[:n_val]
                train_ids = shuffled_train_pool[n_val:]
            else:
                train_ids = train_pool_ids

            metadata = {
                "split_strategy": "family_holdout",
                "seed": seed,
                "test_families": sorted(list(test_fam_set)),
                "val_families": sorted(list(val_fam_set)),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

            return DatasetSplit(
                train_layout_ids=sorted(train_ids),
                val_layout_ids=sorted(val_ids),
                test_layout_ids=sorted(test_ids),
                metadata=metadata,
                layout_to_family=mapping,
            )

        # If test_families is not specified, partition unique families by group ratio
        return LayoutSplitter.split_by_group(
            layout_to_group=mapping,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=seed,
        )
