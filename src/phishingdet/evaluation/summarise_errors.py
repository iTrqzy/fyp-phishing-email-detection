import re

import pandas as pd

from phishingdet.data.loader import repo_root


def make_text_preview(raw_text, max_length=350):
    # Clean and shorten text so it fits nicely in a report.
    # This removes line breaks and repeated spaces, then truncates if needed.

    cleaned_text = str(raw_text).replace("\n", " ").replace("\r", " ").strip()
    cleaned_text = " ".join(cleaned_text.split())

    if len(cleaned_text) > max_length:
        return cleaned_text[:max_length] + "..."

    return cleaned_text


def load_error_analysis_file(csv_path):
    # Load one error analysis CSV if it exists.
    # If the file is missing, return None instead of crashing.

    if not csv_path.exists():
        return None

    return pd.read_csv(csv_path)


def make_deduplication_key(raw_text):
    """
    Create a simplified text key so repeated template-like messages
    collapse into one representative example.

    This helps avoid showing basically the same mistake many times.

    It removes:
    - case differences
    - long numbers / dates / IDs
    - extra whitespace
    """

    # Start from a long cleaned preview version of the text
    cleaned_text = make_text_preview(raw_text, max_length=1000).lower()

    # Replace all digit sequences with the same placeholder
    # Example:
    # "your code is 123456" -> "your code is <num>"
    # This makes template messages with different numbers count as the same pattern
    cleaned_text = re.sub(r"\d+", "<num>", cleaned_text)

    # Normalize whitespace so spacing differences do not matter
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    return cleaned_text


def build_error_section(error_dataframe, error_type_name, top_count=5):
    # Build one text section for a specific error type,
    # e.g. false positives or false negatives.

    # Keep only rows matching the requested error type
    matching_rows = error_dataframe[
        error_dataframe["error_type"].astype(str).str.lower() == error_type_name
    ].copy()

    # If there are no rows for this error type, return a short message
    if matching_rows.empty:
        return [f"No rows found for {error_type_name}.", ""]

    # If phishing_prob exists, make sure it is numeric
    if "phishing_prob" in matching_rows.columns:
        matching_rows["phishing_prob"] = pd.to_numeric(
            matching_rows["phishing_prob"],
            errors="coerce",
        )

        # Sort the errors so the most interesting ones appear first
        #
        # For false positives:
        #   higher phishing_prob means the model was very confident but wrong
        #
        # For false negatives:
        #   lower phishing_prob means the model was very confident in the wrong safe direction
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

    # Create a dedupe key from the text preview
    # This helps collapse repeated template-style errors into one example
    matching_rows["dedupe_key"] = matching_rows["text_preview"].astype(str).apply(
        make_deduplication_key
    )

    # Keep only the first occurrence of each dedupe key
    matching_rows = matching_rows.drop_duplicates(subset="dedupe_key", keep="first")

    # After deduplication, keep only the top requested number
    matching_rows = matching_rows.head(top_count)

    # Start building the text lines for this section
    section_lines = [
        f"{error_type_name.upper()} (Top {len(matching_rows)})",
        "-" * 60,
    ]

    # Add one block per selected error row
    for row_number, (_, row) in enumerate(matching_rows.iterrows(), start=1):
        phishing_probability = row.get("phishing_prob", "")

        # Display probability nicely, or N/A if missing
        if pd.isna(phishing_probability):
            probability_text = "N/A"
        else:
            probability_text = f"{float(phishing_probability):.4f}"

        # Add summary line
        section_lines.append(
            f"{row_number}. "
            f"true={row.get('true_label', 'N/A')} | "
            f"pred={row.get('pred_label', 'N/A')} | "
            f"phishing_prob={probability_text}"
        )

        # Add text preview line
        section_lines.append(make_text_preview(row.get("text_preview", "")))

        # Blank line between examples
        section_lines.append("")

    return section_lines


def write_error_summary(stage_display_name, source_csv_path, output_text_path, top_count=5):
    # Load one stage's error analysis CSV and write a readable text summary.

    error_dataframe = load_error_analysis_file(source_csv_path)

    # If the CSV is missing, write a short message instead
    if error_dataframe is None:
        output_text_path.write_text(
            f"Missing error analysis file for {stage_display_name}: {source_csv_path}\n",
            encoding="utf-8",
        )
        print(f"Missing error file for {stage_display_name}: {source_csv_path}")
        return

    # Build the header of the summary file
    summary_lines = [
        f"Error Summary for {stage_display_name}",
        f"Source CSV: {source_csv_path}",
        f"Total error rows: {len(error_dataframe)}",
        "",
    ]

    # Add false positive section
    summary_lines.extend(
        build_error_section(error_dataframe, "false_positive", top_count=top_count)
    )

    # Add false negative section
    summary_lines.extend(
        build_error_section(error_dataframe, "false_negative", top_count=top_count)
    )

    # Save the summary to a text file
    output_text_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved error summary to: {output_text_path}")


def main(top_count=5):
    # Get project root and make sure artifacts/eval exists
    project_root = repo_root()
    evaluation_output_directory = project_root / "artifacts" / "eval"
    evaluation_output_directory.mkdir(parents=True, exist_ok=True)

    # Input CSV files for Stage 2 and Stage 3 error analysis
    stage2_error_csv_path = project_root / "artifacts" / "stage2_metadata" / "error_analysis.csv"
    stage3_error_csv_path = project_root / "artifacts" / "stage3_hybrid" / "error_analysis.csv"

    # Output text summary files
    stage2_summary_output_path = evaluation_output_directory / "error_summary_stage2.txt"
    stage3_summary_output_path = evaluation_output_directory / "error_summary_stage3.txt"

    # Create Stage 2 text summary
    write_error_summary(
        "Stage 2 (Metadata-only)",
        stage2_error_csv_path,
        stage2_summary_output_path,
        top_count=top_count,
    )

    # Create Stage 3 text summary
    write_error_summary(
        "Stage 3 (Hybrid)",
        stage3_error_csv_path,
        stage3_summary_output_path,
        top_count=top_count,
    )


if __name__ == "__main__":
    # Run the script and keep the top 5 examples per error type
    main(top_count=5)