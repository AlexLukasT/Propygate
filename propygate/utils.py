import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import os


def plot_example(normed_image, prediction, filename="example.png"):
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    y_labels = np.arange(len(prediction))

    font = FontProperties()
    font.set_weight("bold")
    font.set_size("xx-large")

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))

    ax[0].imshow(normed_image, cmap="gray")
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].barh(y_labels, prediction, align="center", edgecolor="black", linewidth=1, height=0.5)
    ax[1].set_xlabel("Probability", fontproperties=font)
    ax[1].set_ylabel("Digit", fontproperties=font)
    ax[1].set_yticks(y_labels)
    ax[1].set_yticklabels(y_labels)
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 1.1)
    ax[1].tick_params(axis='both', which='major', labelsize=20, bottom=True, left=True)

    fig.tight_layout()
    fig.savefig(filename, dpi=150)


def plot_metrics(metrics, filename="metrics.png"):
    if os.path.dirname(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    epochs = np.arange(1, len(metrics["train_loss"]) + 1)

    for metric, values in metrics.items():
        if metric.endswith("loss") and values:
            axes[0].plot(epochs, values, label=metric.replace("_", " "), linewidth=1.5)
        elif metric.endswith("acc") and values:
            axes[1].plot(epochs, values, label=metric.replace("_", " "), linewidth=1.5,
                         linestyle="dashed")
        else:
            continue
    axes[1].set_xlabel("Epoch", fontsize=15)
    axes[0].set_ylabel("Loss [a.u.]", fontsize=15)
    axes[1].set_ylabel("Accuracy [%]", fontsize=15)
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=15)
    fig.tight_layout()
    print(f"Saved metrics plot to {filename}")
    fig.savefig(filename, dpi=150)
