import re

import pandas as pd

from phishingdet.data.loader import load_email, repo_root
from phishingdet.data.splits import get_or_make_split_indices
from phishingdet.features.build_features import transform_vectorizer
from phishingdet.features.build_metadata_features import transform_metadata_vectorizer
from phishingdet.models.predict import load_artifacts as load_stage1_artifacts
from phishingdet.models.predict_hybrid import load_stage3_artifacts


# Fixed seed so the same held-out split is reused consistently
RANDOM_STATE = 42

# Name of the saved split file to reuse
TEST_SPLIT_NAME = "phishing_email_split_1"


def make_text_preview(raw_text, max_length=220):
    # Create a short one-line preview of text for saving in the CSV.
    # This avoids storing huge email bodies in the output file.

    # Convert to string, remove line breaks, and trim outer spaces
    cleaned_text = str(raw_text).replace("\n", " ").replace("\r", " ").strip()

    # Collapse repeated whitespace into single spaces
    cleaned_text = " ".join(cleaned_text.split())

    # Truncate if the text is longer than the allowed preview length
    if len(cleaned_text) > max_length:
        return cleaned_text[:max_length] + "..."

    # Otherwise return the cleaned full preview
    return cleaned_text


def lightly_obfuscate_phishing_text(email_text):
    # Slightly perturb a phishing email to see whether the model still flags it.
    #
    # The idea is not to rewrite the email completely.
    # It only makes small changes such as:
    # - obfuscating a URL
    # - replacing a common phishing phrase with softer wording
    # Then it adds a short prefix at the start.

    email_text = str(email_text)

    # Regex pattern to find a website-like string
    # Examples:
    # - http://example.com
    # - https://example.com
    # - www.example.com
    website_pattern = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)

    def replace_url_with_obfuscated_version(match):
        # Get the matched URL text
        original_url = match.group(0)

        # Obfuscate protocol so it looks less obviously like a clickable URL
        obfuscated_url = original_url.replace("http://", "hxxp://").replace("https://", "hxxps://")

        # Break up dots and slashes slightly
        obfuscated_url = obfuscated_url.replace(".", " [.] ")
        obfuscated_url = obfuscated_url.replace("/", " / ")
        return obfuscated_url

    # If the email contains a URL, obfuscate the first one found
    if website_pattern.search(email_text):
        email_text = website_pattern.sub(
            replace_url_with_obfuscated_version,
            email_text,
            count=1,
        )
    else:
        # If no URL is found, soften one suspicious phrase instead
        replacement_rules = [
            (r"\bclick here\b", "open the link below"),
            (r"\bverify your account\b", "confirm your account"),
            (r"\bupdate your account\b", "refresh your account details"),
            (r"\burgent\b", "important"),
        ]

        # Apply the first rule that actually changes the text
        for search_pattern, replacement_text in replacement_rules:
            updated_text = re.sub(search_pattern, replacement_text, email_text, flags=re.IGNORECASE)
            if updated_text != email_text:
                email_text = updated_text
                break

    # Add a generic prefix so the perturbation is slightly different overall
    return "Important notice: " + email_text


def inject_urgency_into_legitimate_text(email_text):
    # Make a legitimate email sound a bit more suspicious by adding urgency language.
    # This tests whether the model gets too easily influenced by urgent wording.

    email_text = str(email_text)
    return "URGENT: Please review ASAP. " + email_text + " This is time-sensitive but routine."


def score_with_stage1(email_text, stage1_model, stage1_vectorizer, threshold=0.5):
    # Score one email using the Stage 1 text-only model.
    #
    # Returns:
    # - phishing probability
    # - binary class prediction using the given threshold

    # Convert raw text into the text feature representation used by Stage 1
    text_features = transform_vectorizer(stage1_vectorizer, [email_text])

    # Get probability of class 1 = phishing
    phishing_probability = float(stage1_model.predict_proba(text_features)[0][1])

    # Convert probability to a hard class label using the threshold
    binary_prediction = int(phishing_probability >= threshold)

    return phishing_probability, binary_prediction


def score_with_stage3(
    email_text,
    text_model,
    text_vectorizer,
    metadata_model,
    metadata_vectorizer,
    stacking_model,
    threshold=0.5,
):
    # Score one email using the Stage 3 hybrid model.
    #
    # Stage 3 works in three steps:
    # 1. text model gives a phishing probability
    # 2. metadata model gives a phishing probability
    # 3. stacking model combines those two probabilities into one final hybrid probability

    # Transform text for the Stage 3 text model
    text_features = transform_vectorizer(text_vectorizer, [email_text])

    # Get Stage 3 text-model phishing probability
    text_model_probability = float(text_model.predict_proba(text_features)[0][1])

    # Transform text for the metadata model
    metadata_features = transform_metadata_vectorizer(metadata_vectorizer, [email_text])

    # Get metadata-model phishing probability
    metadata_model_probability = float(metadata_model.predict_proba(metadata_features)[0][1])

    # Combine both probabilities into a 2-feature input for the stacking model
    stacked_input_features = [[text_model_probability, metadata_model_probability]]

    # Final hybrid phishing probability
    hybrid_probability = float(stacking_model.predict_proba(stacked_input_features)[0][1])

    # Convert final probability into a hard prediction
    binary_prediction = int(hybrid_probability >= threshold)

    return text_model_probability, metadata_model_probability, hybrid_probability, binary_prediction


def build_small_robustness_dataset():
    # Build a very small robustness benchmark from the held-out test split.
    #
    # Output:
    # - 10 phishing emails, lightly perturbed to make them less obvious
    # - 10 legitimate emails, modified to sound more urgent
    #
    # This gives a small controlled set for checking whether predictions flip
    # after mild wording changes.

    # Load the full cleaned dataset
    full_dataframe = load_email()

    # Extract text and labels as Python lists
    email_texts = full_dataframe["text"].tolist()
    email_labels = full_dataframe["label"].tolist()

    # Reuse the same saved train/test split used elsewhere in the project
    train_indices, test_indices = get_or_make_split_indices(
        email_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=True,
        split_name=TEST_SPLIT_NAME,
    )

    # Build a small DataFrame from only the held-out test rows
    test_rows = [
        {"text": email_texts[index], "label": int(email_labels[index])}
        for index in test_indices
    ]
    test_dataframe = pd.DataFrame(test_rows)

    # Take the first 10 phishing and first 10 legitimate emails from the test split
    phishing_examples = test_dataframe[test_dataframe["label"] == 1].head(10).copy()
    legitimate_examples = test_dataframe[test_dataframe["label"] == 0].head(10).copy()

    # Fail clearly if the test split does not contain enough samples for both groups
    if phishing_examples.empty or legitimate_examples.empty:
        raise ValueError(
            "Not enough phishing and legitimate samples in the test split "
            "to build the robustness mini-benchmark."
        )

    # Mark the phishing perturbation type and create perturbed phishing text
    phishing_examples["perturbation_type"] = "phishing_obfuscated"
    phishing_examples["perturbed_text"] = phishing_examples["text"].apply(
        lightly_obfuscate_phishing_text
    )

    # Mark the legitimate perturbation type and create perturbed legitimate text
    legitimate_examples["perturbation_type"] = "legitimate_urgency_injected"
    legitimate_examples["perturbed_text"] = legitimate_examples["text"].apply(
        inject_urgency_into_legitimate_text
    )

    # Combine both groups into one small benchmark table
    robustness_dataframe = pd.concat(
        [phishing_examples, legitimate_examples],
        ignore_index=True,
    )

    # Add a simple sample id column
    robustness_dataframe["sample_id"] = range(len(robustness_dataframe))

    return robustness_dataframe


def main():
    # Create the evaluation output directory if needed
    output_directory = repo_root() / "artifacts" / "eval"
    output_directory.mkdir(parents=True, exist_ok=True)

    # File where the robustness results will be saved
    output_csv_path = output_directory / "robustness_mini_results.csv"

    # Load Stage 1 text-only model artifacts
    stage1_model, stage1_vectorizer = load_stage1_artifacts()

    # Load Stage 3 hybrid model artifacts
    (
        stage3_text_model,
        stage3_text_vectorizer,
        stage3_metadata_model,
        stage3_metadata_vectorizer,
        stage3_stacking_model,
    ) = load_stage3_artifacts()

    # Build the small robustness benchmark dataset
    robustness_dataframe = build_small_robustness_dataset()

    # List to collect result rows before converting to a DataFrame
    output_rows = []

    # Iterate through each sample in the robustness benchmark
    for _, row in robustness_dataframe.iterrows():
        original_text = row["text"]
        perturbed_text = row["perturbed_text"]

        # Score original and perturbed versions with Stage 1
        stage1_probability_before, stage1_prediction_before = score_with_stage1(
            original_text,
            stage1_model,
            stage1_vectorizer,
        )
        stage1_probability_after, stage1_prediction_after = score_with_stage1(
            perturbed_text,
            stage1_model,
            stage1_vectorizer,
        )

        # Score original and perturbed versions with Stage 3
        _, _, stage3_probability_before, stage3_prediction_before = score_with_stage3(
            original_text,
            stage3_text_model,
            stage3_text_vectorizer,
            stage3_metadata_model,
            stage3_metadata_vectorizer,
            stage3_stacking_model,
        )
        _, _, stage3_probability_after, stage3_prediction_after = score_with_stage3(
            perturbed_text,
            stage3_text_model,
            stage3_text_vectorizer,
            stage3_metadata_model,
            stage3_metadata_vectorizer,
            stage3_stacking_model,
        )

        # Save one result row for this sample
        output_rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "true_label": int(row["label"]),
                "perturbation_type": row["perturbation_type"],
                "original_text_preview": make_text_preview(original_text),
                "perturbed_text_preview": make_text_preview(perturbed_text),

                # Stage 1 before/after scores and predictions
                "stage1_probability_before": stage1_probability_before,
                "stage1_prediction_before": stage1_prediction_before,
                "stage1_probability_after": stage1_probability_after,
                "stage1_prediction_after": stage1_prediction_after,

                # 1 if Stage 1 changed class after perturbation, else 0
                "stage1_prediction_changed": int(stage1_prediction_before != stage1_prediction_after),

                # Stage 3 before/after scores and predictions
                "stage3_probability_before": stage3_probability_before,
                "stage3_prediction_before": stage3_prediction_before,
                "stage3_probability_after": stage3_probability_after,
                "stage3_prediction_after": stage3_prediction_after,

                # 1 if Stage 3 changed class after perturbation, else 0
                "stage3_prediction_changed": int(stage3_prediction_before != stage3_prediction_after),
            }
        )

    # Convert all result rows into a DataFrame
    output_dataframe = pd.DataFrame(output_rows)

    # Save full robustness results to CSV
    output_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8")

    # Calculate flip rate:
    # the proportion of samples whose prediction changed after perturbation
    stage1_flip_rate = float(output_dataframe["stage1_prediction_changed"].mean())
    stage3_flip_rate = float(output_dataframe["stage3_prediction_changed"].mean())

    # Print summary
    print("Robustness mini-benchmark complete.")
    print("Number of samples:", len(output_dataframe))
    print("Stage 1 prediction flip rate:", round(stage1_flip_rate, 3))
    print("Stage 3 prediction flip rate:", round(stage3_flip_rate, 3))
    print("Saved results to:", output_csv_path)


if __name__ == "__main__":
    # Run the mini robustness experiment
    main()