"""
Import and dataset preprocessing module
"""
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DiabetesDataset(Dataset):
    """
    PyTorch Dataset for the diabetes risk.

    It reads the CSV file, codifies categorical features by one-hot encoding
    and converts it to tensors.

    :param csv_path: CSV file route. If it is None, it uses data/raw route.
    :type csv_path: Path or str, optional
    """

    def __init__(self, csv_path: Path | str | None = None):
        if csv_path is None:
            project_root = Path(__file__).resolve().parents[1]
            csv_path = project_root / "data" / "raw" / "diabetes_risk.csv"

        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find dataset in: {self.csv_path}")

        # Cargar y preprocesar
        df = pd.read_csv(self.csv_path)
        X_df = df.iloc[:, :-1]
        y_series = df.iloc[:, -1]

        # One-hot encoding para variables categóricas
        X_df = pd.get_dummies(X_df, drop_first=True)
        self.X = torch.tensor(X_df.values.astype(np.float32), dtype=torch.float32)

        # Conversión de etiquetas a códigos numéricos consecutivos
        y_cat = y_series.astype("category")
        self.y = torch.tensor(y_cat.cat.codes.values, dtype=torch.long)

        self.num_features = self.X.shape[1]
        self.num_classes = len(y_cat.cat.categories)

    def __len__(self) -> int:
        """Returns total number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a tuple (characteristics, label) for the given index"""
        return self.X[idx], self.y[idx]


def get_dataloader(batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, int, int]:
    """
    Utility function for instantiating the Dataset and returning its DataLoader ready to train.

    :param batch_size: Batch size.
    :param shuffle: If data must be shuffled
    :return: Tuple with (DataLoader, num_features, num_classes).
    """
    dataset = DiabetesDataset()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, dataset.num_features, dataset.num_classes