"""
Module for evaluate model and save the graphic reports.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from practice_sdoml.dataset import get_dataloader
from practice_sdoml.modeling.train import train
from practice_sdoml.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_top_loss_samples,
)


def evaluate_and_generate_figures():
    reports_dir = Path(__file__).resolve().parents[2] / "reports" / "figures"
    train_loader, _, num_classes = get_dataloader(batch_size=32)

    model = train(epochs=15)
    model.eval()

    all_preds, all_targets, all_probs, sample_losses = [], [], [], []
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            losses = criterion_none(outputs, batch_y)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            sample_losses.extend(losses.numpy())
            all_preds.extend(preds.numpy())
            all_targets.extend(batch_y.numpy())
            all_probs.extend(probs.numpy())

    targets = np.array(all_targets)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    losses = np.array(sample_losses)

    # Delegación de trazado al módulo plots
    plot_confusion_matrix(targets, preds, reports_dir / "confusion_matrix.png")
    plot_top_loss_samples(losses, reports_dir / "top_loss_samples.png")
    plot_calibration_curve(targets, probs, num_classes, reports_dir / "calibration_curve.png")
    print(f"Figures generated in: {reports_dir}")


if __name__ == "__main__":
    evaluate_and_generate_figures()