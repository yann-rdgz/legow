import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
from matplotlib.transforms import Bbox


TIME_UNTI = {"day": 1, "month": 30, "year": 365}


class SurvivalAnalysis:
    def __init__(self, df):
        self.df = df
        self.survival_summary = []

    def survival_analysis(
        self,
        event_col,
        time_col,
        covariate_columns=None,
        verbose=True,
        univariate=True,
        **cox_kwargs,
    ):
        """Given a dataset designed for survival analysis, returns a summary with univariate or multivariate statistics:
            - HR with confidence interval and associated p value
            - C-index
        Continuous variables are Z-centered
        Non binary categorical variables are ignored

        Parameters
        ----------
        event_col : [str], optional
            Column associated to the event variable (bool)
        time_col : [str], optional
            Column associated to the time to event or censoring variable (float)
        covariate_columns: List[str], optional
            List of columns used as covariates in Cox survival models. If None, allcolumns except event_col and
            time_col
        univariate: [bool], optional
            Whether to perform Cox regression per variable or for all variables

        Returns
        -------
        summary DataFrame
        """

        # Get summary dataset
        df = self.df.copy()
        n_tot = len(df)
        covariate_columns = covariate_columns or df.columns.tolist()
        summary = pd.DataFrame()

        for col in covariate_columns:
            # Check if binary or continuous
            n_values = df[col].nunique()
            if col in [event_col, time_col]:
                continue
            elif n_values == 2:
                column_type = "bin."
            elif (df[col].dtype in [str]) or (n_values == 1):
                if verbose:
                    print(f"Removing {col}")
            else:
                column_type = "cont."

            # Add summary information
            n = df[col].notnull().sum()
            col_results = {
                "variable": col,
                "type": column_type,
                "is_univariate": univariate,
                "counts": n,
                "missing": f"{100*(n_tot-n)/n_tot:.0f}%",
            }

            if column_type == "bin.":
                values, counts = zip(*df[col].value_counts().items(), strict=False)
                try:
                    v = max(values)
                except ValueError:
                    # Set the most common value to True
                    v = values[0]
                df[col] = df[col] == v
                col_results["value"] = v
                col_results["mean"] = df[col].mean()
                col_results["std"] = None
            else:
                col_results["value"] = None
                col_results["mean"] = df[col].mean()
                col_results["std"] = df[col].std()
                # Z-center continuous variables
                df[col] = (df[col] - df[col].mean()) / df[col].std()

            if univariate:
                mask = df[col].notnull()
                df_uni = df.loc[mask, [col, time_col, event_col]]
                cox_model = CoxPHFitter(**cox_kwargs).fit(df_uni, time_col, event_col)

                # Add cox results in the row
                for k in [
                    "exp(coef)",
                    "exp(coef) lower 95%",
                    "exp(coef) upper 95%",
                    "p",
                ]:
                    col_results[k] = cox_model.summary.loc[col, k]
                c_index = cox_model.concordance_index_
                c_index = max(c_index, 1 - c_index)
                col_results["C_index"] = c_index
                p_PH = proportional_hazard_test(cox_model, df_uni[[col, event_col, time_col]]).p_value[0] > 0.01
                col_results["PH"] = p_PH
            pd.concat([summary, col_results])
        summary = summary.set_index("variable")

        if not univariate:
            columns = covariate_columns + [c for c in (event_col, time_col) if c not in covariate_columns]
            df = df[columns]
            # Median imputation
            df = df.fillna(df.median(), axis=0)
            cox_model = CoxPHFitter(**cox_kwargs).fit(df, time_col, event_col)
            covariates = cox_model.summary.index.values
            for k in ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]:
                summary.at[covariates, k] = cox_model.summary[k]
            c_index = cox_model.concordance_index_
            c_index = max(c_index, 1 - c_index)
            summary["C_index"] = c_index
            summary["PH"] = proportional_hazard_test(cox_model, df).p_value > 0.01

        summary.rename(
            {
                "exp(coef)": "HR",
                "exp(coef) lower 95%": "HR low",
                "exp(coef) upper 95%": "HR high",
            },
            axis=1,
            inplace=True,
        )
        summary = summary.sort_values(by="p", ascending=True)
        self.survival_summary.append(summary)
        return summary

    def forest_plot(
        self,
        summary=None,
        figsize=None,
        ratio_col="HR",
        ratio_low_col="HR low",
        ratio_high_col="HR high",
        xlim=None,
        row_colors=("#E5E8E8", "#FDFEFE"),
        title="Hazard ratio",
        minimal=False,
    ):
        """Plot a forest plot

        Parameters
        ----------
        summary: pd.DataFrame
            Univariate Cox model summary, as output from survival_analysis. If default (None), use
            self.survival_summary[-1]
        figsize : [tuple], optional
            figsize, by default None
        ratio_col : str, optional
            name of the column to use for the ratio, by default 'HR'
        ratio_low_col : str, optional
            name of the column to use for the lower value of the ratio, by default 'HR low'
        ratio_high_col : str, optional
            name of the column to use for the higher value of the ratio, by default 'HR low'
        xlim : [tuple], optional
            xlim, by default None
        row_colors : list / tuple, optional
            colors for the rows, by default ['#E5E8E8', '#FDFEFE']
        title : str, optional
            title, by default 'Hazard ratio'

        Returns
        -------
        None
        """
        if summary is None:
            if len(self.survival_summary) == 0:
                raise ValueError("`survival_summary` attribute is empty. Call `survival_analysis` method first")
            summary = self.survival_summary[-1]

        if not summary["is_univariate"].all():
            raise ValueError("Forest plot only applies to univariate analyses.")

        n = len(summary)
        summary = summary.copy()

        # ==== Create figure ====
        if figsize is None:
            figsize = (6, 0.7 * n) if minimal else (10, 0.7 * n)
        fig, ax = plt.subplots(figsize=figsize)

        # Remove borders
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.margins(y=0)

        # Color rows
        row_colors = list(row_colors)
        for i in range(n):
            ax.axhspan(i - 0.5, i + 0.5, facecolor=row_colors[i % len(row_colors)])

        # Other
        ax.grid()
        ax.set_yticks([])
        ax.set_title(title)
        if xlim is not None:
            ax.set_xlim(xlim)

        # ==== Plot error bars ====
        ratios = summary[ratio_col]
        ratios_low = summary[ratio_low_col]
        ratios_high = summary[ratio_high_col]

        # Get color for each error bar associated to the p value
        color_list = ["green", "lightgreen", "orange", "red"]

        def get_error_color(p):
            color_index = np.sum(p >= np.array([0.01 / n, 0.05 / n, 0.05]))
            return color_list[color_index]

        error_colors = [get_error_color(p) for p in summary["p"]]

        # Add color legend
        for color, label in zip(*[color_list, ["p < 0.01/n", "p < 0.05/n", "p < 0.05", "p>= 0.05"]], strict=False):
            ax.plot([], [], "-o", c=color, label=label)
        ax.legend()

        # Draw line for ratio = 1
        ax.axvline(1, color="k")

        # Draw error bars
        ax.errorbar(
            x=ratios,
            y=np.arange(n)[::-1],
            xerr=[ratios - ratios_low, ratios_high - ratios],
            fmt=" ",
            ecolor=tuple(error_colors),
            lw=2,
        )

        # Draw ratio values
        ax.scatter(ratios, np.arange(n)[::-1], marker="o", color=error_colors, s=30, zorder=2)

        # ==== TABLE ====
        del summary[ratio_low_col]
        del summary[ratio_high_col]
        if minimal:
            summary = summary[["type", "HR", "p", "C_index"]]

        rowColours = np.array((row_colors * n)[:n][::-1])
        cellColours = rowColours[:, None].repeat(summary.shape[1], 1)
        for col in summary.columns:
            if summary[col].dtype == float:
                summary[col] = summary[col].apply(lambda x: f"{x:.2f}" if isinstance(x, float) else x)

        table_size = 0.4 if minimal else 0.6
        _ = ax.table(
            cellText=summary.values,
            rowLabels=pd.Series(summary.index).apply(lambda x: x[:20]),
            rowColours=tuple(rowColours),
            cellColours=tuple(cellColours),
            colLabels=summary.columns,
            cellLoc="center",
            loc="left",
            bbox=Bbox.from_bounds(-table_size, 0.0, table_size, (n + 1) / n),
        )
        # for cell in table._cells:
        #     if cell[0] == 0:
        #         table._cells[cell].get_text().set_rotation(45)

        # # Font change
        # table.auto_set_font_size(False)

        # # Removing table borders
        # for key, cell in table.get_celld().items():
        #     cell.set_linewidth(0)
