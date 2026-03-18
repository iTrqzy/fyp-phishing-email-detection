import re

import pandas as pd

from phishingdet.data.loader import repo_root


def make_text_preview(raw_text, max_length=350):
    # Clean up and shorten a text snippet so it is easy to inspect in a report.
    cleaned_text = str(raw_text).replace("\n", " ").replace("\r", " ").strip()
    cleaned_text = " ".join(cleaned_text.split())

    if len(cleaned_text) > max_length:
        return cleaned_text[:max_length] + "..."

    return cleaned_text


def load_error_analysis_file(csv_path):
    # Return a DataFrame if the file exists, otherwise return None.
    if not csv_path.exists():
        return None

    return pd.read_csv(csv_path)


def make_deduplication_key(raw_text):
    """
    Create a simplified text key so repeated template-like messages
    collapse into a single representative example.

    This removes:
    - case differences
    - long digit sequences / dates / IDs
    - extra whitespace
    """
    cleaned_text = make_text_preview(raw_text, max_length=1000).lower()

    # Replace numbers with a placeholder so messages that only differ
    # by dates, hours, IDs or tracking numbers count as the same pattern.
    cleaned_text = re.sub(r"\d+", "<num>", cleaned_text)

    # Normalise whitespace
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def build_error_section(error_dataframe, error_type_name, top_count=5):
    # Create one text section for either false positives or false negatives.
    matching_rows = error_dataframe[
        error_dataframe["error_type"].astype(str).str.lower() == error_type_name
    ].copy()

    if matching_rows.empty:
        return [f"No rows found for {error_type_name}.", ""]

    # Make sure phishing_prob is numeric if present
    if "phishing_prob" in matching_rows.columns:
        matching_rows["phishing_prob"] = pd.to_numeric(
            matching_rows["phishing_prob"],
            errors="coerce",
        )

        # Sort so the most interesting mistakes appear first
        if error_type_name == "false_positive":
            matching_rows = matching_rows.sort_values(
                "phishing_prob",
                ascending=False,
                na_position="last",
            )
        else:
            matching_rows = matching_rows.sort_values(
                "phishing_prob",
                ascending=True,
                na_position="last",
            )

    # Deduplicate repeated template-style errors using a normalised preview key
    matching_rows["dedupe_key"] = matching_rows["text_preview"].astype(str).apply(
        make_deduplication_key
    )
    matching_rows = matching_rows.drop_duplicates(subset="dedupe_key", keep="first")

    # Only keep the requested number after deduplication
    matching_rows = matching_rows.head(top_count)

    section_lines = [
        f"{error_type_name.upper()} (Top {len(matching_rows)})",
        "-" * 60,
    ]

    for row_number, (_, row) in enumerate(matching_rows.iterrows(), start=1):
        phishing_probability = row.get("phishing_prob", "")
        if pd.isna(phishing_probability):
            probability_text = "N/A"
        else:
            probability_text = f"{float(phishing_probability):.4f}"

        section_lines.append(
            f"{row_number}. "
            f"true={row.get('true_label', 'N/A')} | "
            f"pred={row.get('pred_label', 'N/A')} | "
            f"phishing_prob={probability_text}"
        )
        section_lines.append(make_text_preview(row.get("text_preview", "")))
        section_lines.append("")

    return section_lines


def write_error_summary(stage_display_name, source_csv_path, output_text_path, top_count=5):
    # Read one error analysis CSV and write a report-friendly text summary.
    error_dataframe = load_error_analysis_file(source_csv_path)

    if error_dataframe is None:
        output_text_path.write_text(
            f"Missing error analysis file for {stage_display_name}: {source_csv_path}\n",
            encoding="utf-8",
        )
        print(f"Missing error file for {stage_display_name}: {source_csv_path}")
        return

    summary_lines = [
        f"Error Summary for {stage_display_name}",
        f"Source CSV: {source_csv_path}",
        f"Total error rows: {len(error_dataframe)}",
        "",
    ]

    summary_lines.extend(
        build_error_section(error_dataframe, "false_positive", top_count=top_count)
    )
    summary_lines.extend(
        build_error_section(error_dataframe, "false_negative", top_count=top_count)
    )

    output_text_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved error summary to: {output_text_path}")


def main(top_count=5):
    project_root = repo_root()
    evaluation_output_directory = project_root / "artifacts" / "eval"
    evaluation_output_directory.mkdir(parents=True, exist_ok=True)

    stage2_error_csv_path = project_root / "artifacts" / "stage2_metadata" / "error_analysis.csv"
    stage3_error_csv_path = project_root / "artifacts" / "stage3_hybrid" / "error_analysis.csv"

    stage2_summary_output_path = evaluation_output_directory / "error_summary_stage2.txt"
    stage3_summary_output_path = evaluation_output_directory / "error_summary_stage3.txt"

    write_error_summary(
        "Stage 2 (Metadata-only)",
        stage2_error_csv_path,
        stage2_summary_output_path,
        top_count=top_count,
    )

    write_error_summary(
        "Stage 3 (Hybrid)",
        stage3_error_csv_path,
        stage3_summary_output_path,
        top_count=top_count,
    )


if __name__ == "__main__":
    main(top_count=5)