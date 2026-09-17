from pathlib import Path
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder

# Import SimpleNet
from .model import SimpleNet

def load_data(batch_size: int = 32):
    """
    Loads diabetes risk dataset and it prepare it in Pytorch DataLoaders

    Reads CSV file from raw data directory, splits the characteristics of the target,
    encodes the cathegorical variables and transforms data in tensors.

    :param batch_size: Batch size for DataLoader. Default size is 32.
    :type batch_size: int
    :return: A tuple containing:
            - train_loader (DataLoader): Training data iterator.
            - num_features (int): Number of input features.
            - num_classes (int): Number of classes to predict.
    :rtype: tuple
    """

    raw_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "diabetes_risk.csv"
    df = pd.read_csv(raw_path)

    # Split X and y
    X_df = df.iloc[:, :-1]
    y_series = df.iloc[:, -1]

    # one-hot encoding
    X_df = pd.get_dummies(X_df, drop_first=True)
    X = X_df.values.astype(np.float32)

    # Coversion
    y_categorical = y_series.astype("category")
    y = y_categorical.cat.codes.values
    num_classes = len(y_categorical.cat.categories)

    # 2. Conversion to PyTorch vectors
    x_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True), X.shape[1], num_classes



def train(epochs: int = 3, lr: float = 0.001):
    """
    Executes training loop for the neural network SimpleNet

    Instances model, defines loss function (CrossEntropyLoss) and optimizer (Adam) and 
    repeates through the training data for the specified epoch number, printing the loss
    in each epoch to verify learning.

    :param epochs: Number of loops through the training dataset.
    :type epochs: int
    :param lr: Learning rate for the Adam optimizer.
    :type lr: float
    :return: Neural network model trained.
    :rtype: SimpleNet
    """
    
    train_loader, num_features, num_classes = load_data()

    # Instance the net pre-defined
    model = SimpleNet(input_dim=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("--- Initializing training ---")

    # Training Loop: uploads parameters and shows the progress
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        # Verifying that the model is learning
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("--- Training finished ---")
    return model

if __name__ == "__main__":
    train()