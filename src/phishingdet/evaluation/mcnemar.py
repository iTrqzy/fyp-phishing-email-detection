import pandas as pd
from scipy.stats import binomtest, chi2

from phishingdet.data.loader import repo_root


def load_stage_predictions(stage_name):
    # Load one stage prediction file from artifacts/eval.
    csv_path = repo_root() / "artifacts" / "eval" / f"{stage_name}_test_preds.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing prediction CSV: {csv_path}")

    prediction_dataframe = pd.read_csv(csv_path)
    return prediction_dataframe, csv_path

def calculate_mcnemar_chi_square(stage1_correct_stage3_wrong, stage1_wrong_stage3_correct):
    # Continuity-corrected McNemar chi-square statistic and p-value.

    disagreement_count = stage1_correct_stage3_wrong + stage1_wrong_stage3_correct

    if disagreement_count == 0:
        return 0.0, 1.0

    chi_square_statistic = (
        (abs(stage1_correct_stage3_wrong - stage1_wrong_stage3_correct) - 1) ** 2
    ) / disagreement_count

    p_value = float(chi2.sf(chi_square_statistic, df=1))
    return float(chi_square_statistic), p_value


def calculate_exact_binomial_p_value(stage1_correct_stage3_wrong, stage1_wrong_stage3_correct):
    # Exact two-sided binomial test for McNemar disagreements.

    disagreement_count = stage1_correct_stage3_wrong + stage1_wrong_stage3_correct

    if disagreement_count == 0:
        return 1.0

    smaller_disagreement_count = min(
        stage1_correct_stage3_wrong,
        stage1_wrong_stage3_correct,
    )

    exact_test_result = binomtest(
        smaller_disagreement_count,
        n=disagreement_count,
        p=0.5,
        alternative="two-sided",
    )

    return float(exact_test_result.pvalue)

def main(prediction_column_name="pred_0_5"):
    output_directory = repo_root() / "artifacts" / "eval"
    output_directory.mkdir(parents=True, exist_ok=True)

    report_path = output_directory / "mcnemar_stage1_vs_stage3.txt"

    stage1_dataframe, stage1_csv_path = load_stage_predictions("stage1")
    stage3_dataframe, stage3_csv_path = load_stage_predictions("stage3")

    required_columns = {"id", "y_true", prediction_column_name}

    for stage_name, prediction_dataframe, csv_path in [
        ("stage1", stage1_dataframe, stage1_csv_path),
        ("stage3", stage3_dataframe, stage3_csv_path),
    ]:
        missing_columns = required_columns - set(prediction_dataframe.columns)
        if missing_columns:
            raise ValueError(
                f"{csv_path.name} is missing columns: {sorted(missing_columns)}"
            )

    if len(stage1_dataframe) != len(stage3_dataframe):
        raise ValueError(
            "Stage 1 and Stage 3 prediction files do not contain the same number of rows."
        )

    # Sort both files by id so the same test samples line up exactly
    stage1_dataframe = stage1_dataframe.sort_values("id").reset_index(drop=True)
    stage3_dataframe = stage3_dataframe.sort_values("id").reset_index(drop=True)

    stage1_true_labels = stage1_dataframe["y_true"].astype(int)
    stage3_true_labels = stage3_dataframe["y_true"].astype(int)

    if not stage1_true_labels.equals(stage3_true_labels):
        raise ValueError(
            "Stage 1 and Stage 3 y_true columns do not match. "
            "Re-run both models on the same held-out split."
        )

    true_labels = stage1_true_labels
    stage1_predictions = stage1_dataframe[prediction_column_name].astype(int)
    stage3_predictions = stage3_dataframe[prediction_column_name].astype(int)

    stage1_is_correct = stage1_predictions.eq(true_labels)
    stage3_is_correct = stage3_predictions.eq(true_labels)

    both_models_correct = int((stage1_is_correct & stage3_is_correct).sum())
    stage1_correct_stage3_wrong = int((stage1_is_correct & ~stage3_is_correct).sum())
    stage1_wrong_stage3_correct = int((~stage1_is_correct & stage3_is_correct).sum())
    both_models_wrong = int((~stage1_is_correct & ~stage3_is_correct).sum())

    chi_square_statistic, chi_square_p_value = calculate_mcnemar_chi_square(
        stage1_correct_stage3_wrong,
        stage1_wrong_stage3_correct,
    )
    exact_binomial_p_value = calculate_exact_binomial_p_value(
        stage1_correct_stage3_wrong,
        stage1_wrong_stage3_correct,
    )

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

    if exact_binomial_p_value < 0.05:
        report_lines.append(
            "Interpretation: the difference is statistically significant at the 5% level."
        )
    else:
        report_lines.append(
            "Interpretation: the difference is not statistically significant at the 5% level."
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n".join(report_lines))
    print()
    print("Saved McNemar report to:", report_path)


if __name__ == "__main__":
    main(prediction_column_name="pred_0_5")