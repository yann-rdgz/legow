import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency, norm


class PowerAnalysis:
    """
    A class for performing power analysis for logrank and chi-square tests.
    """

    def __init__(self, alpha=0.05, power_threshold=0.85):
        """
        Initialize the PowerAnalysis object.

        Parameters:
            alpha (float): Significance level for statistical tests.
            power_threshold (float): Desired statistical power threshold.
        """
        self.alpha = alpha
        self.power_threshold = power_threshold

    def compute_power_logrank(self, n_patients, hazard_ratio, event_rate):
        """
        Computes the power for a logrank test.

        Parameters:
            n_patients (int): Number of patients.
            hazard_ratio (float): Hazard ratio to test.
            event_rate (float): Event rate in the population.

        Returns:
            float: Statistical power.
        """
        n_events = n_patients * event_rate
        z_alpha = norm.ppf(1 - self.alpha / 2)
        log_hr = np.log(hazard_ratio)
        se_log_hr = np.sqrt(4 / n_events)
        z = log_hr / se_log_hr
        power = norm.cdf(z - z_alpha) + norm.cdf(-z - z_alpha)
        return power

    def compute_power_simulation(self, n_patients, response_rate_control, response_rate_treatment, n_simulations=10000):
        """
        Simulates power for a chi-square test using Monte Carlo simulation.

        Parameters:
            n_patients (int): Total number of patients.
            response_rate_control (float): Response rate in the control group.
            response_rate_treatment (float): Response rate in the treatment group.
            n_simulations (int): Number of simulations to run.

        Returns:
            float: Simulated statistical power.
        """
        n_control = n_patients // 2
        n_treatment = n_patients - n_control
        power_count = 0

        for _ in range(n_simulations):
            control_group = np.random.binomial(1, response_rate_control, n_control)
            treatment_group = np.random.binomial(1, response_rate_treatment, n_treatment)

            contingency_table = np.array(
                [
                    [np.sum(control_group), n_control - np.sum(control_group)],
                    [np.sum(treatment_group), n_treatment - np.sum(treatment_group)],
                ]
            )
            _, p_value, _, _ = chi2_contingency(contingency_table)

            if p_value < self.alpha:
                power_count += 1

        return power_count / n_simulations

    def plot_power_analysis(self, n_patients_range, metrics, test_type="logrank"):
        """
        Plots the power analysis.

        Parameters:
            n_patients_range (array-like): Range of patient numbers.
            metrics (list): A list of tuples containing (value, label).
                            For logrank, provide hazard ratios.
                            For chi-square, provide response rate pairs.
            test_type (str): Type of test ('logrank' or 'chi-square').

        Returns:
            matplotlib.figure.Figure: The resulting plot figure.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axhline(
            self.power_threshold,
            color="black",
            linestyle="--",
            label=f"{self.power_threshold * 100:.0f}% Power Threshold",
        )
        color_palette = plt.get_cmap("Set1")(np.linspace(0, 1, 8))
        for i, metric in enumerate(metrics):
            powers = []
            for n_patients in n_patients_range:
                if test_type == "logrank":
                    power = self.compute_power_logrank(n_patients, metric[0], event_rate=0.4)
                elif test_type == "chi-square":
                    power = self.compute_power_simulation(n_patients, metric[0][0], metric[0][1])
                else:
                    raise ValueError(f"Unknown test type: {test_type}")
                powers.append(power)

            ax.plot(n_patients_range, powers, label=metric[1], color=color_palette[i % len(color_palette)])

            intercept_idx = np.argmax(np.array(powers) >= self.power_threshold)
            if intercept_idx < len(n_patients_range):
                intercept_patients = n_patients_range[intercept_idx]
                ax.plot(
                    intercept_patients,
                    self.power_threshold,
                    "o",
                    label=f"N={intercept_patients}",
                    color=color_palette[i % len(color_palette)],
                    alpha=0.7,
                )

        ax.set_xlabel("Number of Patients")
        ax.set_ylabel("Statistical Power")
        ax.set_title(f"Power Analysis ({test_type.capitalize()} Test)")
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.grid(True)
        return fig

    @staticmethod
    def combine_hazard_ratio(frac_1, hr_1, hr_2):
        """
        Combines two hazard ratios based on the fraction of patients in each group.

        Parameters:
            frac_1 (float): Fraction of the first group.
            hr_1 (float): Hazard ratio for the first group.
            hr_2 (float): Hazard ratio for the second group.

        Returns:
            float: Combined hazard ratio.
        """
        frac_2 = 1 - frac_1
        return np.exp(frac_1 * np.log(hr_1) + frac_2 * np.log(hr_2))

    @staticmethod
    def combine_response_rate(frac_1, rr_1, rr_2):
        """
        Combines two response rates based on the fraction of patients in each group.

        Parameters:
            frac_1 (float): Fraction of the first group.
            rr_1 (float): Response rate for the first group.
            rr_2 (float): Response rate for the second group.

        Returns:
            float: Combined response rate.
        """
        frac_2 = 1 - frac_1
        return frac_1 * rr_1 + frac_2 * rr_2


# Example usage
if __name__ == "__main__":
    analysis = PowerAnalysis(alpha=0.05, power_threshold=0.85)

    # Logrank example
    hazard_ratios = [(0.7, "Initial population"), (0.6, "Enriched population")]
    n_patients_range = np.arange(50, 1000, 1)
    fig = analysis.plot_power_analysis(n_patients_range, hazard_ratios, test_type="logrank")
    plt.show()

    # Chi-square example
    response_rates = [((0.239, 0.403), "Baseline vs Combo"), ((0.30, 0.60), "Endotype Comparison")]
    n_patients_range = np.arange(100, 800, 10)
    fig = analysis.plot_power_analysis(n_patients_range, response_rates, test_type="chi-square")
    plt.show()
