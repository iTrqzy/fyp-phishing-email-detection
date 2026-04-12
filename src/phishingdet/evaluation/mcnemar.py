import pandas as pd
from scipy.stats import binomtest, chi2

from phishingdet.data.loader import repo_root


def load_stage_predictions(stage_name):
    csv_path = repo_root() / "artifacts" / "eval" / f"{stage_name}_test_preds.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    prediction_dataframe = pd.read_csv(csv_path)

    return prediction_dataframe, csv_path


def calculate_mcnemar_chi_square(stage1_correct_stage3_wrong, stage1_wrong_stage3_correct):
    # McNemar focuses only on the disagreement counts:
    # - cases where Stage 1 is correct and Stage 3 is wrong
    # - cases where Stage 1 is wrong and Stage 3 is correct

    # Total number of disagreements between the two models
    disagreement_count = stage1_correct_stage3_wrong + stage1_wrong_stage3_correct

    # If there are no disagreements at all, the models behaved identically
    # so the statistic is 0 and p-value is 1
    if disagreement_count == 0:
        return 0.0, 1.0

    # Continuity-corrected McNemar chi-square formula
    # The "-1" is the continuity correction
    # When the models disagree, is one model winning much more often than the other
    chi_square_statistic = (
        (abs(stage1_correct_stage3_wrong - stage1_wrong_stage3_correct) - 1) ** 2
    ) / disagreement_count

    # Convert the chi-square statistic into a p-value using the chi-square distribution
    p_value = float(chi2.sf(chi_square_statistic, df=1))

    return float(chi_square_statistic), p_value


def calculate_exact_binomial_p_value(stage1_correct_stage3_wrong, stage1_wrong_stage3_correct):
    # Calculate the exact two-sided binomial p-value for the McNemar disagreements.
    # This is often preferred when the disagreement counts are small.

    # Total number of disagreements
    disagreement_count = stage1_correct_stage3_wrong + stage1_wrong_stage3_correct

    # If there are no disagreements, then there is no evidence of a difference
    if disagreement_count == 0:
        return 1.0

    # For the exact test, use the smaller of the two disagreement counts
    smaller_disagreement_count = min(
        stage1_correct_stage3_wrong,
        stage1_wrong_stage3_correct,
    )

    # Run a two-sided binomial test assuming p = 0.5 under the null hypothesis
    # Null hypothesis: both models are equally likely to win on a disagreement
    exact_test_result = binomtest(
        smaller_disagreement_count,
        n=disagreement_count,
        p=0.5,
        alternative="two-sided",
    )

    # Return the p-value from the exact test
    return float(exact_test_result.pvalue)


def main(prediction_column_name="pred_0_5"):
    # Build the evaluation output directory
    output_directory = repo_root() / "artifacts" / "eval"
    output_directory.mkdir(parents=True, exist_ok=True)

    # File where the McNemar report will be saved
    report_path = output_directory / "mcnemar_stage1_vs_stage3.txt"

    # Load Stage 1 and Stage 3 prediction CSVs
    stage1_dataframe, stage1_csv_path = load_stage_predictions("stage1")
    stage3_dataframe, stage3_csv_path = load_stage_predictions("stage3")

    # These columns must exist in both files:
    # - id: sample id
    # - y_true: true label
    # - prediction_column_name: prediction to compare, e.g. pred_0_5
    required_columns = {"id", "y_true", prediction_column_name}

    # Check each file has the required columns
    for stage_name, prediction_dataframe, csv_path in [
        ("stage1", stage1_dataframe, stage1_csv_path),
        ("stage3", stage3_dataframe, stage3_csv_path),
    ]:
        missing_columns = required_columns - set(prediction_dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path.name} is missing columns: {sorted(missing_columns)}"
            )

    # Both files must have the same number of rows
    if len(stage1_dataframe) != len(stage3_dataframe):
        raise ValueError(
            "Stage 1 and Stage 3 prediction files do not contain the same number of rows."
        )

    # Sort both files by id so the same test samples line up in the same order
    stage1_dataframe = stage1_dataframe.sort_values("id").reset_index(drop=True)
    stage3_dataframe = stage3_dataframe.sort_values("id").reset_index(drop=True)

    # Extract the true labels from both files
    stage1_true_labels = stage1_dataframe["y_true"].astype(int)
    stage3_true_labels = stage3_dataframe["y_true"].astype(int)

    # Confirm both files really refer to the exact same test set
    if not stage1_true_labels.equals(stage3_true_labels):
        raise ValueError(
            "Stage 1 and Stage 3 y_true columns do not match. "
            "Re-run both models on the same held-out split."
        )

    # Since they match, use either one as the true labels
    true_labels = stage1_true_labels

    # Extract the chosen prediction column from both files
    stage1_predictions = stage1_dataframe[prediction_column_name].astype(int)
    stage3_predictions = stage3_dataframe[prediction_column_name].astype(int)

    # Compare each model's predictions against the true labels
    # This gives a True/False value for each sample
    stage1_is_correct = stage1_predictions.eq(true_labels)
    stage3_is_correct = stage3_predictions.eq(true_labels)

    # Build the 2x2 McNemar contingency counts
    both_models_correct = int((stage1_is_correct & stage3_is_correct).sum())
    stage1_correct_stage3_wrong = int((stage1_is_correct & ~stage3_is_correct).sum())
    stage1_wrong_stage3_correct = int((~stage1_is_correct & stage3_is_correct).sum())
    both_models_wrong = int((~stage1_is_correct & ~stage3_is_correct).sum())

    # Calculate the approximate McNemar chi-square result
    chi_square_statistic, chi_square_p_value = calculate_mcnemar_chi_square(
        stage1_correct_stage3_wrong,
        stage1_wrong_stage3_correct,
    )

    # Calculate the exact binomial McNemar p-value
    exact_binomial_p_value = calculate_exact_binomial_p_value(
        stage1_correct_stage3_wrong,
        stage1_wrong_stage3_correct,
    )

    # Build the text report line by line
    report_lines = [
        "McNemar Test: Stage 1 vs Stage 3",
        f"Prediction column used: {prediction_column_name}",
        "",
        f"Stage 1 CSV: {stage1_csv_path}",
        f"Stage 3 CSV: {stage3_csv_path}",
        "",
        "Contingency table:",
        "                         Stage 3 correct   Stage 3 wrong",
        f"Stage 1 correct          {both_models_correct:<16d}{stage1_correct_stage3_wrong}",
        f"Stage 1 wrong            {stage1_wrong_stage3_correct:<16d}{both_models_wrong}",
        "",
        f"Stage 1 correct, Stage 3 wrong: {stage1_correct_stage3_wrong}",
        f"Stage 1 wrong, Stage 3 correct: {stage1_wrong_stage3_correct}",
        "",
        f"Continuity-corrected chi-square statistic: {chi_square_statistic:.6f}",
        f"Approximate p-value: {chi_square_p_value:.6f}",
        f"Exact binomial p-value: {exact_binomial_p_value:.6f}",
        "",
    ]

    # Add a plain-English interpretation using the usual 5% significance threshold
    if exact_binomial_p_value < 0.05:
        report_lines.append(
            "Interpretation: the difference is statistically significant at the 5% level."
        )
    else:
        report_lines.append(
            "Interpretation: the difference is not statistically significant at the 5% level."
        )

    # Save the report to a text file
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    # Also print the report to the console
    print("\n".join(report_lines))
    print()
    print("Saved McNemar report to:", report_path)


if __name__ == "__main__":
    # Run the comparison using the default threshold-based predictions
    main(prediction_column_name="pred_0_5")