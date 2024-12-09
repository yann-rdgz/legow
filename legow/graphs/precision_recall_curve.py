import matplotlib.pyplot as plt
import numpy as np


class PrecisionRecallVisualizer:
    """
    A class for visualizing Precision-Recall curves with optional iso F1-score lines.

    Attributes:
        color (str): Default color for the PR curve.
    """

    def visualize(self, results, fig=None, axis=None, label=None):
        """
        Visualize a Precision-Recall curve with optional confidence intervals
        and annotated thresholds.

        Args:
            results (dict): A dictionary where keys are thresholds and values are dictionaries
                            containing "precision" and "recall" values.
            fig (matplotlib.figure.Figure, optional): An existing Matplotlib figure. Defaults to None.
            axis (matplotlib.axes.Axes, optional): An existing Matplotlib axis. Defaults to None.
            label (str, optional): Label for the PR curve in the legend. Defaults to None.

        Returns:
            tuple: A tuple (fig, axis) with the figure and axis objects used for the plot.
        """
        multiple_experiments = isinstance(results[list(results.keys())[0]]["precision"], list)
        thresholds = sorted(results.keys())
        precisions = [results[t]["precision"] for t in thresholds]
        recalls = [results[t]["recall"] for t in thresholds]

        if multiple_experiments:
            precisions_std = [np.std(p) for p in precisions]
            precisions = [np.mean(p) for p in precisions]
            recalls = [np.mean(r) for r in recalls]

        if fig is None or axis is None:
            fig, axis = plt.subplots(1, 1, figsize=(8, 6))

        self._add_iso_f1_lines(axis, 0.05)
        axis.plot(recalls, precisions, marker="o", label=label)

        if multiple_experiments:
            axis.fill_between(
                recalls,
                np.array(precisions) - np.array(precisions_std),
                np.array(precisions) + np.array(precisions_std),
                alpha=0.2,
            )

        for i, txt in enumerate(thresholds):
            axis.annotate(
                f"{txt}",
                (recalls[i], precisions[i]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        axis.legend()
        axis.set_xlim([0.1, 1])
        axis.set_ylim([0.1, 1])
        axis.set_xlabel("Recall (Sensitivity)")
        axis.set_ylabel("Precision")
        axis.set_title("Precision-Recall Curve with iso F1-scores")
        return fig, axis

    @staticmethod
    def _add_iso_f1_lines(axes, axis_start):
        """
        Add iso F1-score contour lines to a plot.

        Args:
            axes (matplotlib.axes.Axes): The Matplotlib axis on which to add the contour lines.
            axis_start (float): The starting value for the iso F1-score levels.

        Returns:
            matplotlib.axes.Axes: Updated axis with iso F1-score lines.
        """
        axis_start = max(axis_start, 0.05)  # Prevent axis from starting at 0
        x_contour, y_contour = np.meshgrid(*[np.arange(0.1, 1.1, 0.1)] * 2)
        z_contour = 2 * x_contour * y_contour / (x_contour + y_contour)

        contour_lines = axes.contour(
            x_contour[0],
            y_contour[:, 0],
            z_contour,
            levels=np.arange(axis_start, 1.1, 0.05),
            colors="gray",
            linestyles=":",
            linewidths=0.5,
        )

        axes.clabel(contour_lines, contour_lines.levels, inline=True, fontsize=8, fmt="%1.2f")
        return axes
