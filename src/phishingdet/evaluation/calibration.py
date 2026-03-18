import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from phishingdet.data.loader import repo_root


STAGE_FILES = {
    "stage1": "stage1_test_preds.csv",
    "stage2": "stage2_test_preds.csv",
    "stage3": "stage3_test_preds.csv",
}

def calculate_expected_calibration_error(true_labels, predicted_probabilities, number_of_bins=10):

    # Calculate Expected Calibration Error (ECE).
    # ECE measures how close the predicted probabilities are to the
    # actual observed frequencies.

    true_labels = np.asarray(true_labels, dtype=int)
    predicted_probabilities = np.asarray(predicted_probabilities, dtype=float)

    bin_edges = np.linspace(0.0, 1.0, number_of_bins + 1)
    probability_bin_indices = np.digitize(predicted_probabilities, bin_edges[1:-1], right=True)

    expected_calibration_error_value = 0.0
    per_bin_summary = []

    for calibration_bin_index in range(number_of_bins):
        in_this_bin = probability_bin_indices == calibration_bin_index
        samples_in_bin = int(in_this_bin.sum())

        if samples_in_bin == 0:
            per_bin_summary.append(
                {
                    "bin_index": int(calibration_bin_index),
                    "sample_count": 0,
                    "average_predicted_probability": None,
                    "observed_positive_rate": None,
                    "absolute_difference": None,
                }
            )
            continue

        average_predicted_probability = float(predicted_probabilities[in_this_bin].mean())
        observed_positive_rate = float(true_labels[in_this_bin].mean())
        absolute_difference = abs(observed_positive_rate - average_predicted_probability)

        expected_calibration_error_value += (samples_in_bin / len(predicted_probabilities)) * absolute_difference

        per_bin_summary.append(
            {
                "bin_index": int(calibration_bin_index),
                "sample_count": samples_in_bin,
                "average_predicted_probability": average_predicted_probability,
                "observed_positive_rate": observed_positive_rate,
                "absolute_difference": absolute_difference,
            }
        )

    return float(expected_calibration_error_value), per_bin_summary

def load_prediction_file(csv_filename):
    # Load one prediction CSV from artifacts/eval.

    csv_path = repo_root() / "artifacts" / "eval" / csv_filename

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    prediction_dataframe = pd.read_csv(csv_path)

    required_columns = {"y_true", "prob"}
    missing_columns = required_columns - set(prediction_dataframe.columns)

    if missing_columns:
        raise ValueError(
            f"{csv_path.name} is missing columns: {sorted(missing_columns)}"
        )

    return prediction_dataframe, csv_path

def main():
    output_directory = repo_root() / "artifacts" / "eval"
    output_directory.mkdir(parents=True, exist_ok=True)

    calibration_plot_path = output_directory / "calibration.png"
    calibration_metrics_path = output_directory / "calibration_metrics.json"

    calibration_results = {}

    # Creating the plots
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")

    for stage_name, csv_filename in STAGE_FILES.items():
        prediction_dataframe, csv_path = load_prediction_file(csv_filename)

        true_labels = prediction_dataframe["y_true"].astype(int).to_numpy()
        predicted_probabilities = prediction_dataframe["prob"].astype(float).to_numpy()

        brier_score_value = float(brier_score_loss(true_labels, predicted_probabilities))

        expected_calibration_error_value, per_bin_summary = calculate_expected_calibration_error(
            true_labels,
            predicted_probabilities,
            number_of_bins=10,
        )

        observed_positive_rates, average_predicted_probabilities = calibration_curve(
            true_labels,
            predicted_probabilities,
            n_bins=10,
            strategy="uniform",
        )

        calibration_results[stage_name] = {
            "source_csv": str(csv_path),
            "sample_count": int(len(prediction_dataframe)),
            "brier_score": brier_score_value,
            "expected_calibration_error": expected_calibration_error_value,
            "per_bin_summary": per_bin_summary,
        }

        plt.plot(
            average_predicted_probabilities,
            observed_positive_rates,
            marker="o",
            linewidth=2,
            label=(
                f"{stage_name.upper()} "
                f"(Brier={brier_score_value:.4f}, "
                f"ECE={expected_calibration_error_value:.4f})"
            ),
        )

    plt.title("Calibration / Reliability Curve (Test Set)")
    plt.xlabel("Average Predicted Probability")
    plt.ylabel("Observed Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(calibration_plot_path, dpi=200)
    plt.close()

    calibration_metrics_path.write_text(
        json.dumps(calibration_results, indent=2),
        encoding="utf-8",
    )

    print("Saved calibration plot to:", calibration_plot_path)
    print("Saved calibration metrics to:", calibration_metrics_path)


if __name__ == "__main__":
    main()