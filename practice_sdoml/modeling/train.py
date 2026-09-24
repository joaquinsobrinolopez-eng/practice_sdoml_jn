"""
Training module for SimpleNet.
"""
import torch
import torch.nn as nn
import torch.optim as optim

from practice_sdoml.dataset import get_dataloader
from practice_sdoml.modeling.model import SimpleNet


def train(epochs: int = 3, lr: float = 0.001):
    """
    Trains SimpleNet model using provided data by DiabetesDataset
    """
    train_loader, num_features, num_classes = get_dataloader(batch_size=32)

    model = SimpleNet(input_dim=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("--- Initializing training ---")
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
        print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss:.4f}")

    print("--- Training finished ---")
    return model


if __name__ == "__main__":
    train()