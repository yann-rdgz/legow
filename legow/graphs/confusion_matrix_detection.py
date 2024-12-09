import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec


class ConfusionMatrixDetection:
    """Confusion Matrix for detection tasks.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Confusion matrix.
    false_positives : np.ndarray
        False positives for each class.
    false_negatives : np.ndarray
        False negatives for each class.
    labels : list
        List of labels for the classes.
    title : str, optional
        Title of the plot (default is 'Confusion Matrix').
    """

    def __init__(
        self,
        confusion_matrix: np.ndarray,
        false_positives: np.ndarray,
        false_negatives: np.ndarray,
        labels: list,
        title="Confusion Matrix",
    ):
        if len(confusion_matrix) != len(labels) != len(false_positives) != len(false_negatives):
            raise ValueError("all inputs must have the same length")

        self.confusion_matrix = confusion_matrix
        self.false_positives = false_positives
        self.false_negatives = false_negatives
        self.labels = labels
        self.title = title

    def plot(self, fig=None):
        if not fig:
            fig = plt.figure(figsize=(8, 8))

        gs = gridspec.GridSpec(
            3,
            3,
            width_ratios=[3 * len(self.labels), 1, 2],
            height_ratios=[3 * len(self.labels), 1, 2],
            wspace=0.05,
            hspace=0.05,
        )

        cm = self.confusion_matrix.T

        precision_lvl1 = cm.sum(axis=1) / (cm.sum(axis=1) + np.array(self.false_positives))
        precision_lvl2 = np.diag(cm) / (cm.sum(axis=1) + np.array(self.false_positives))
        precision = np.stack([precision_lvl1, precision_lvl2])

        recall_lvl1 = cm.sum(axis=0) / (cm.sum(axis=0) + np.array(self.false_negatives))
        recall_lvl2 = np.diag(cm) / (cm.sum(axis=0) + np.array(self.false_negatives))
        recall = np.stack([recall_lvl1, recall_lvl2])

        # Plot the confusion matrix heatmap
        vmin = min(np.min(cm), np.min(self.false_positives), np.min(self.false_negatives))
        vmax = max(np.max(cm), np.max(self.false_positives), np.max(self.false_negatives))

        ax0 = plt.subplot(gs[0, 0])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.labels,
            yticklabels=self.labels,
            cbar=False,
            ax=ax0,
            vmin=vmin,
            vmax=vmax,
        )
        ax0.set_xlabel("Ground Truth")
        ax0.set_ylabel("Prediction")
        ax0.xaxis.set_ticks_position("top")
        ax0.xaxis.set_label_position("top")

        # Plot precision as a separate bar chart
        ax1 = plt.subplot(gs[0, 1])
        sns.heatmap(
            np.array(self.false_positives).reshape(-1, 1),
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            xticklabels=["FP"],
            yticklabels=[],
            ax=ax1,
            vmin=vmin,
            vmax=vmax,
        )
        ax1.set_ylabel("")
        ax1.xaxis.set_ticks_position("top")
        ax1.xaxis.set_label_position("top")

        ax2 = plt.subplot(gs[0, 2])
        sns.heatmap(
            np.array(precision).T,
            annot=True,
            fmt=".2f",
            cmap="magma_r",
            cbar=False,
            xticklabels=["lvl 1", "lvl 2"],
            yticklabels=[],
            ax=ax2,
            vmin=0.2,
            vmax=0.9,
        )
        ax2.set_xlabel("Precision")
        ax2.xaxis.set_ticks_position("top")
        ax2.xaxis.set_label_position("top")

        # Plot recall as a separate bar chart
        ax3 = plt.subplot(gs[1, 0])
        sns.heatmap(
            np.array(self.false_negatives).reshape(1, -1),
            annot=True,
            fmt=".0f",
            cmap="Blues",
            cbar=False,
            yticklabels=["FN"],
            xticklabels=[],
            ax=ax3,
            vmin=vmin,
            vmax=vmax,
        )
        ax3.set_ylabel("")

        ax4 = plt.subplot(gs[2, 0])
        sns.heatmap(
            np.array(recall),
            annot=True,
            fmt=".2f",
            cmap="magma_r",
            cbar=False,
            xticklabels=[],
            yticklabels=["lvl1", "lvl2"],
            ax=ax4,
            vmin=0.2,
            vmax=0.9,
        )
        ax4.set_ylabel("Recall")
        ax4.set_xlabel("")

        plt.suptitle(self.title)

        # Show the plot
        return fig
