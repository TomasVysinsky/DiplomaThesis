from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch

from DataLoader import DataLoaderProprietary
from DataViewModel import TestVehicleSplitComand
from WindowedLeBasedDataset import WindowedLeBasedDataset


FEATURE_NAMES = [
    "axis_x_acc",
    "axis_y_acc",
    "axis_z_acc",
    "sig_pwr",
    "mask",
]

CLASS_NAMES = ["L", "R", "B1", "B2"]


@dataclass
class SplitDataContext:
    dataset_stem: str
    loaded_data_view_model: object
    train_dataset: WindowedLeBasedDataset
    test_dataset: WindowedLeBasedDataset
    feature_names: List[str]
    class_names: List[str]
    split_commands: List[TestVehicleSplitComand]


def resolve_dataset_stem(dataset_path: str) -> str:
    """
    User may pick either:
    - .../final_dataset_nove_vozidla_2
    - .../final_dataset_nove_vozidla_2_vehicles.pcl
    - .../final_dataset_nove_vozidla_2_train_vehicles.pcl
    - .../final_dataset_nove_vozidla_2_test_vehicles.pcl

    Internally the loader expects the stem without suffix.
    """
    p = Path(dataset_path)
    name = p.name

    suffixes = [
        "_vehicles.pcl",
        "_train_vehicles.pcl",
        "_test_vehicles.pcl",
        "_vehicles.joblib",
        "_train_vehicles.joblib",
        "_test_vehicles.joblib",
    ]

    for suffix in suffixes:
        if name.endswith(suffix):
            return str(p.with_name(name[: -len(suffix)]))

    return str(p)


def build_default_split_commands(loaded_data_view_model) -> List[TestVehicleSplitComand]:
    """
    Default strategy for MVP:
    - all manually annotated / 'video' vehicles -> test
    - everything else -> train

    If no 'video' vehicle exists, fallback to the last vehicle as test so that
    test_dataset is not empty.
    """
    vehicles = loaded_data_view_model.vehicles
    video_ecvs = [v.ecv for v in vehicles if "video" in v.ecv]

    if video_ecvs:
        return [TestVehicleSplitComand(1.0, ecv) for ecv in video_ecvs]

    if len(vehicles) == 0:
        raise ValueError("No vehicles found in loaded dataset.")

    return [TestVehicleSplitComand(1.0, vehicles[-1].ecv)]


def load_split_context(
    dataset_path: str,
    window_size: int = 50,
    max_duration_seconds: float = 25,
    normalize_values: bool = True,
    split_commands: List[TestVehicleSplitComand] | None = None,
) -> SplitDataContext:
    dataset_stem = resolve_dataset_stem(dataset_path)

    print("dataset_path =", dataset_path)
    print("dataset_stem =", dataset_stem)

    loaded = DataLoaderProprietary.load_all_data(
        load_from_csv_files=False,
        file_name=dataset_stem,
        load_video_annotations=True,
    )

    split_commands = split_commands or build_default_split_commands(loaded)

    # Important: this keeps your original split-based normalization pipeline
    loaded.split_to_train_and_test(split_commands, normalize_values=normalize_values)

    train_dataset = WindowedLeBasedDataset(
        vehicles=loaded.train_vehicles,
        window_size=window_size,
        name="train",
        max_dlzka_trvania_le=max_duration_seconds,
    )

    test_dataset = WindowedLeBasedDataset(
        vehicles=loaded.test_vehicles,
        window_size=window_size,
        name="test",
        max_dlzka_trvania_le=max_duration_seconds,
    )

    return SplitDataContext(
        dataset_stem=dataset_stem,
        loaded_data_view_model=loaded,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        feature_names=FEATURE_NAMES,
        class_names=CLASS_NAMES,
        split_commands=split_commands,
    )


def get_dataset(context: SplitDataContext, split: str = "test") -> WindowedLeBasedDataset:
    split = split.lower()
    if split == "train":
        return context.train_dataset
    if split == "test":
        return context.test_dataset
    raise ValueError("split must be 'train' or 'test'")


def get_dataset_size(context: SplitDataContext, split: str = "test") -> int:
    dataset = get_dataset(context, split=split)
    return len(dataset)


def get_sample(
    context: SplitDataContext,
    sample_idx: int,
    split: str = "test",
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = get_dataset(context, split=split)

    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(f"sample_idx={sample_idx} out of range [0, {len(dataset) - 1}] for split='{split}'")

    sample, label = dataset[sample_idx]
    return sample, label