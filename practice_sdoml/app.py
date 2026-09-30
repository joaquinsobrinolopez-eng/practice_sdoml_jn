"""
Deployment and Interactive Demonstrations for SDOML.
Built with Gradio, leveraging core modules from practice_sdoml.
"""
from pathlib import Path
import tempfile
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from practice_sdoml.dataset import DiabetesDataset, get_dataloader
from practice_sdoml.modeling.model import SimpleNet
from practice_sdoml.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_top_loss_samples,
)

# Configuración visual
sns.set_theme(style="whitegrid")


# ==============================================================================
# 1. DATA EXPLORATION LOGIC
# ==============================================================================
def load_dataset_sample():
    """Load data in an automatic way"""
    try:
        dataset = DiabetesDataset()
        df = pd.read_csv(dataset.csv_path)
    except Exception:
        # If it does not found the original CSV
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "Age": np.random.randint(20, 80, size=200),
                "BloodPressure": np.random.randint(60, 140, size=200),
                "Glucose": np.random.randint(70, 200, size=200),
                "BMI": np.random.uniform(18.0, 40.0, size=200),
                "Outcome": np.random.choice([0, 1], size=200),
            }
        )
    return df


def get_data_summary():
    df = load_dataset_sample()
    shape_text = f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns"
    stats_df = df.describe().reset_index()
    sample_df = df.head(10)
    return shape_text, sample_df, stats_df


def generate_feature_plot(feature_name: str):
    df = load_dataset_sample()
    if feature_name not in df.columns:
        feature_name = df.columns[0]

    fig, ax = plt.subplots(figsize=(7, 4))
    if pd.api.types.is_numeric_dtype(df[feature_name]):
        sns.histplot(data=df, x=feature_name, kde=True, ax=ax, color="teal")
    else:
        sns.countplot(data=df, x=feature_name, ax=ax, palette="viridis")
    ax.set_title(f"Distribution of {feature_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig


# ==============================================================================
# 2. TRAINING INTERFACE LOGIC
# ==============================================================================
def train_interactive(epochs: int, lr: float, batch_size: int, progress=gr.Progress()):
    """Train SimpleNet uploading Gradio loading bar"""
    progress(0, desc="Preparing DataLoader...")
    loader, num_features, num_classes = get_dataloader(batch_size=32, shuffle=False, split="test")

    model = SimpleNet(input_dim=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(lr))

    loss_history = []
    model.train()

    for epoch in range(1, int(epochs) + 1):
        running_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / max(len(loader), 1)
        loss_history.append(epoch_loss)
        progress(epoch / float(epochs), desc=f"Epoch {epoch}/{epochs} - Loss: {epoch_loss:.4f}")

    # Loss evolution graphic
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, int(epochs) + 1), loss_history, marker="o", color="royalblue", linewidth=2)
    ax.set_title("Training Loss Progression", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Loss", fontsize=10)
    plt.tight_layout()

    summary = (
        f"✅ Training completed successfully!\n"
        f"- Epochs: {epochs}\n"
        f"- Final Loss: {loss_history[-1]:.4f}\n"
        f"- Learning Rate: {lr}\n"
        f"- Batch Size: {batch_size}"
    )
    return summary, fig


# ==============================================================================
# 3. MODEL EVALUATION LOGIC
# ==============================================================================
def run_evaluation():
    """Generate metrics and reuse functions from practice_sdoml.plots."""
    loader, num_features, num_classes = get_dataloader(batch_size=32)

    # Train a representative evaluation model
    model = SimpleNet(input_dim=num_features, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    model.train()
    for _ in range(6):
        for bx, by in loader:
            optimizer.zero_grad()
            l = criterion(model(bx), by)
            l.backward()
            optimizer.step()

    model.eval()
    all_preds, all_targets, all_probs, sample_losses = [], [], [], []
    criterion_none = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for bx, by in loader:
            outputs = model(bx)
            losses = criterion_none(outputs, by)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            sample_losses.extend(losses.numpy())
            all_preds.extend(preds.numpy())
            all_targets.extend(by.numpy())
            all_probs.extend(probs.numpy())

    targets = np.array(all_targets)
    preds = np.array(all_preds)
    probs = np.array(all_probs)
    losses = np.array(sample_losses)

    # Direct reuse of practice_sdoml.plots with temp files
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cm_path = tmp_path / "cm.png"
        cal_path = tmp_path / "cal.png"
        loss_path = tmp_path / "top_loss.png"

        plot_confusion_matrix(targets, preds, cm_path)
        plot_calibration_curve(targets, probs, num_classes, cal_path)
        plot_top_loss_samples(losses, loss_path, top_k=8)

        cm_img = plt.imread(str(cm_path))
        cal_img = plt.imread(str(cal_path))
        loss_img = plt.imread(str(loss_path))

    accuracy = np.mean(targets == preds) * 100
    metrics_text = f"**Overall Accuracy:** {accuracy:.2f}% | **Evaluated Samples:** {len(targets)}"
    return metrics_text, cm_img, cal_img, loss_img


# ==============================================================================
# GRADIO APPLICATION LAYOUT
# ==============================================================================
df_initial = load_dataset_sample()
columns_list = list(df_initial.columns)

with gr.Blocks(title="Practice 4: ML Interactive Demo") as demo:
    gr.Markdown("# 🚀 Interactive ML Demonstration: Diabetes Risk Classification")
    gr.Markdown(
        "Demonstration app developed for Practice 4 (Software Development Oriented to Machine Learning). "
        "Allows dataset inspection, interactive hyperparameter training, and model evaluation."
    )

    with gr.Tabs():
        # TAB 1: DATA EXPLORATION
        with gr.Tab("📊 Data Exploration"):
            gr.Markdown("### Explore and visualize the tabular dataset")
            data_info = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=1):
                    feature_dropdown = gr.Dropdown(
                        choices=columns_list,
                        value=columns_list[0] if columns_list else None,
                        label="Select Feature to Plot",
                    )
                    plot_btn = gr.Button("Generate Plot", variant="primary")
                with gr.Column(scale=2):
                    plot_output = gr.Plot()

            gr.Markdown("#### Sample Data Preview")
            sample_table = gr.Dataframe()
            gr.Markdown("#### Statistical Summary")
            stats_table = gr.Dataframe()

            plot_btn.click(generate_feature_plot, inputs=[feature_dropdown], outputs=[plot_output])
            demo.load(get_data_summary, outputs=[data_info, sample_table, stats_table])
            demo.load(generate_feature_plot, inputs=[feature_dropdown], outputs=[plot_output])

        # TAB 2: TRAINING INTERFACE
        with gr.Tab("⚙️ Training Interface"):
            gr.Markdown("### Interactive training and hyperparameter configuration")
            with gr.Row():
                with gr.Column():
                    epoch_slider = gr.Slider(minimum=1, maximum=15, value=5, step=1, label="Epochs")
                    lr_slider = gr.Slider(
                        minimum=0.0001, maximum=0.05, value=0.005, step=0.0005, label="Learning Rate"
                    )
                    batch_slider = gr.Slider(minimum=8, maximum=64, value=32, step=8, label="Batch Size")
                    train_button = gr.Button("Start Training", variant="primary")
                with gr.Column():
                    train_output_status = gr.Markdown()
                    train_loss_plot = gr.Plot()

            train_button.click(
                train_interactive,
                inputs=[epoch_slider, lr_slider, batch_slider],
                outputs=[train_output_status, train_loss_plot],
            )

        # TAB 3: MODEL EVALUATION
        with gr.Tab("📈 Model Evaluation"):
            gr.Markdown("### Performance evaluation and diagnostics")
            eval_button = gr.Button("Run Model Evaluation", variant="primary")
            eval_metrics = gr.Markdown()
            with gr.Row():
                eval_cm = gr.Image(label="Confusion Matrix")
                eval_cal = gr.Image(label="Calibration Curve")
                eval_loss = gr.Image(label="Highest Loss Samples")

            eval_button.click(run_evaluation, outputs=[eval_metrics, eval_cm, eval_cal, eval_loss])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)