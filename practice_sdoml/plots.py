"""
Graphic utilities for model evaluation module
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve


def plot_confusion_matrix(targets: np.ndarray, preds: np.ndarray, output_path: Path) -> None:
    """
    Creates and saves confusion matrix

    :param targets: True labels.
    :param preds: Model predictions.
    :param output_path: FIle route for saving images.
    """
    plt.figure(figsize=(6, 5))
    cm = confusion_matrix(targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca())
    plt.title("Confusion Matrix - Diabetes Risk Model", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_top_loss_samples(sample_losses: np.ndarray, output_path: Path, top_k: int = 10) -> None:
    """
    Creates a barplot with the worst classifications.

    :param sample_losses: Loss array per sample.
    :param output_path: FIle route for saving images.
    :param top_k: Sample number to show.
    """
    plt.figure(figsize=(8, 4))
    worst_indices = np.argsort(sample_losses)[-top_k:]
    worst_losses = sample_losses[worst_indices]

    sns.barplot(x=[f"Sample {i}" for i in worst_indices], y=worst_losses, color="crimson")
    plt.title(f"Top {top_k} Samples with Highest Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Sample Index", fontsize=10)
    plt.ylabel("Loss Value", fontsize=10)
    plt.xticks(rotation=45)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_calibration_curve(targets: np.ndarray, probs: np.ndarray, num_classes: int, output_path: Path) -> None:
    """
    Creates and saves fiability diagram.

    :param targets: True labels.
    :param probs: Predicted probabilities.
    :param num_classes: Class number.
    :param output_path: FIle route for saving images.
    """
    plt.figure(figsize=(6, 5))
    target_binary = (targets == 1).astype(int)
    prob_pos = probs[:, 1] if num_classes > 1 else probs[:, 0]

    fraction_of_positives, mean_predicted_value = calibration_curve(
        target_binary, prob_pos, n_bins=5, strategy="uniform"
    )

    plt.plot(mean_predicted_value, fraction_of_positives, "s-", color="magenta", label="Model Calibration")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.title("Model Calibration Analysis (Reliability Diagram)", fontsize=12, fontweight="bold")
    plt.xlabel("Mean Predicted Probability", fontsize=10)
    plt.ylabel("Fraction of Positives", fontsize=10)
    plt.legend(loc="upper left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()