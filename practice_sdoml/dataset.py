"""
Data ingestion and preprocessing module with train/test partitioning support.
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class DiabetesDataset(Dataset):
    """
    PyTorch Dataset for diabetes_risk.csv supporting stratified train/test splits.

    :param csv_path: Path to the raw CSV file. Defaults to data/raw/diabetes_risk.csv.
    :type csv_path: Path | str | None
    :param split: Target data partition ('train', 'test', or 'all').
    :type split: str
    :param test_size: Fraction of samples allocated to the test subset.
    :type test_size: float
    :param random_state: Random seed for deterministic reproducibility.
    :type random_state: int
    """

    def __init__(
        self,
        csv_path: Path | str | None = None,
        split: str = "train",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        if csv_path is None:
            project_root = Path(__file__).resolve().parents[1]
            csv_path = project_root / "data" / "raw" / "diabetes_risk.csv"

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {self.csv_path}")

        # Load raw tabular data
        df = pd.read_csv(self.csv_path)
        X_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]

        # Apply one-hot encoding to categorical features
        X_df = pd.get_dummies(X_df, drop_first=True)
        X_all = X_df.values.astype(np.float32)

        # Convert categorical target labels to zero-indexed integer encodings
        y_cat = y_series.astype("category")
        y_all = y_cat.cat.codes.values.astype(np.int64)

        self.num_features = X_all.shape[1]
        self.num_classes = len(y_cat.cat.categories)

        # Perform stratified train/test split if requested
        if split in ["train", "test"]:
            X_train, X_test, y_train, y_test = train_test_split(
                X_all,
                y_all,
                test_size=test_size,
                random_state=random_state,
                stratify=y_all,
            )
            if split == "train":
                self.X = torch.tensor(X_train, dtype=torch.float32)
                self.y = torch.tensor(y_train, dtype=torch.long)
            else:
                self.X = torch.tensor(X_test, dtype=torch.float32)
                self.y = torch.tensor(y_test, dtype=torch.long)
        else:
            self.X = torch.tensor(X_all, dtype=torch.float32)
            self.y = torch.tensor(y_all, dtype=torch.long)

    def __len__(self) -> int:
        """Returns the total number of samples in the active partition."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves feature and label tensors for a given sample index.

        :param idx: Sample index.
        :type idx: int
        :return: Tuple containing (features_tensor, label_tensor).
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        return self.X[idx], self.y[idx]


def get_dataloader(
    batch_size: int = 32,
    shuffle: bool = True,
    split: str = "train",
) -> Tuple[DataLoader, int, int]:
    """
    Factory helper function to instantiate DiabetesDataset and build a DataLoader.

    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param shuffle: Whether to shuffle samples at every epoch.
    :type shuffle: bool
    :param split: Partition identifier ('train', 'test', or 'all').
    :type split: str
    :return: Tuple containing (DataLoader, num_features, num_classes).
    :rtype: Tuple[DataLoader, int, int]
    """
    dataset = DiabetesDataset(split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset.num_features, dataset.num_classes