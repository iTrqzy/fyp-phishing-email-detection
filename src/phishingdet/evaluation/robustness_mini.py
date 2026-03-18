import re

import pandas as pd

from phishingdet.data.loader import load_email, repo_root
from phishingdet.data.splits import get_or_make_split_indices
from phishingdet.features.build_features import transform_vectorizer
from phishingdet.features.build_metadata_features import transform_metadata_vectorizer
from phishingdet.models.predict import load_artifacts as load_stage1_artifacts
from phishingdet.models.predict_hybrid import load_stage3_artifacts


RANDOM_STATE = 42
TEST_SPLIT_NAME = "phishing_email_split_1"


def make_text_preview(raw_text, max_length=220):
    # Shorten text for saving in the CSV.

    cleaned_text = str(raw_text).replace("\n", " ").replace("\r", " ").strip()
    cleaned_text = " ".join(cleaned_text.split())

    if len(cleaned_text) > max_length:
        return cleaned_text[:max_length] + "..."

    return cleaned_text


def lightly_obfuscate_phishing_text(email_text):
    # Make a phishing email slightly harder by obfuscating links or softening

    email_text = str(email_text)

    website_pattern = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)

    def replace_url_with_obfuscated_version(match):
        original_url = match.group(0)
        obfuscated_url = original_url.replace("http://", "hxxp://").replace("https://", "hxxps://")
        obfuscated_url = obfuscated_url.replace(".", " [.] ")
        obfuscated_url = obfuscated_url.replace("/", " / ")
        return obfuscated_url

    if website_pattern.search(email_text):
        email_text = website_pattern.sub(
            replace_url_with_obfuscated_version,
            email_text,
            count=1,
        )
    else:
        replacement_rules = [
            (r"\bclick here\b", "open the link below"),
            (r"\bverify your account\b", "confirm your account"),
            (r"\bupdate your account\b", "refresh your account details"),
            (r"\burgent\b", "important"),
        ]

        for search_pattern, replacement_text in replacement_rules:
            updated_text = re.sub(search_pattern, replacement_text, email_text, flags=re.IGNORECASE)
            if updated_text != email_text:
                email_text = updated_text
                break

    return "Important notice: " + email_text


def inject_urgency_into_legitimate_text(email_text):
    # Make a legitimate email more suspicious by adding urgency language.

    email_text = str(email_text)
    return "URGENT: Please review ASAP. " + email_text + " This is time-sensitive but routine."


def score_with_stage1(email_text, stage1_model, stage1_vectorizer, threshold=0.5):
    # Get Stage 1 probability and binary prediction.

    text_features = transform_vectorizer(stage1_vectorizer, [email_text])
    phishing_probability = float(stage1_model.predict_proba(text_features)[0][1])
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

    # Get Stage 3 hybrid probability and binary prediction.
    text_features = transform_vectorizer(text_vectorizer, [email_text])
    text_model_probability = float(text_model.predict_proba(text_features)[0][1])

    metadata_features = transform_metadata_vectorizer(metadata_vectorizer, [email_text])
    metadata_model_probability = float(metadata_model.predict_proba(metadata_features)[0][1])

    stacked_input_features = [[text_model_probability, metadata_model_probability]]
    hybrid_probability = float(stacking_model.predict_proba(stacked_input_features)[0][1])
    binary_prediction = int(hybrid_probability >= threshold)

    return text_model_probability, metadata_model_probability, hybrid_probability, binary_prediction


def build_small_robustness_dataset():
    # Create a tiny benchmark from the held-out test split:
    # - 10 phishing emails, lightly obfuscated
    # - 10 legitimate emails, urgency injected

    full_dataframe = load_email()
    email_texts = full_dataframe["text"].tolist()
    email_labels = full_dataframe["label"].tolist()

    train_indices, test_indices = get_or_make_split_indices(
        email_labels,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=True,
        split_name=TEST_SPLIT_NAME,
    )

    test_rows = [
        {"text": email_texts[index], "label": int(email_labels[index])}
        for index in test_indices
    ]
    test_dataframe = pd.DataFrame(test_rows)

    phishing_examples = test_dataframe[test_dataframe["label"] == 1].head(10).copy()
    legitimate_examples = test_dataframe[test_dataframe["label"] == 0].head(10).copy()

    if phishing_examples.empty or legitimate_examples.empty:
        raise ValueError(
            "Not enough phishing and legitimate samples in the test split "
            "to build the robustness mini-benchmark."
        )

    phishing_examples["perturbation_type"] = "phishing_obfuscated"
    phishing_examples["perturbed_text"] = phishing_examples["text"].apply(
        lightly_obfuscate_phishing_text
    )

    legitimate_examples["perturbation_type"] = "legitimate_urgency_injected"
    legitimate_examples["perturbed_text"] = legitimate_examples["text"].apply(
        inject_urgency_into_legitimate_text
    )

    robustness_dataframe = pd.concat(
        [phishing_examples, legitimate_examples],
        ignore_index=True,
    )
    robustness_dataframe["sample_id"] = range(len(robustness_dataframe))

    return robustness_dataframe


def main():
    output_directory = repo_root() / "artifacts" / "eval"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_csv_path = output_directory / "robustness_mini_results.csv"

    # Load the trained model artifacts
    stage1_model, stage1_vectorizer = load_stage1_artifacts()

    (
        stage3_text_model,
        stage3_text_vectorizer,
        stage3_metadata_model,
        stage3_metadata_vectorizer,
        stage3_stacking_model,
    ) = load_stage3_artifacts()

    robustness_dataframe = build_small_robustness_dataset()
    output_rows = []

    for _, row in robustness_dataframe.iterrows():
        original_text = row["text"]
        perturbed_text = row["perturbed_text"]

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

        output_rows.append(
            {
                "sample_id": int(row["sample_id"]),
                "true_label": int(row["label"]),
                "perturbation_type": row["perturbation_type"],
                "original_text_preview": make_text_preview(original_text),
                "perturbed_text_preview": make_text_preview(perturbed_text),
                "stage1_probability_before": stage1_probability_before,
                "stage1_prediction_before": stage1_prediction_before,
                "stage1_probability_after": stage1_probability_after,
                "stage1_prediction_after": stage1_prediction_after,
                "stage1_prediction_changed": int(stage1_prediction_before != stage1_prediction_after),
                "stage3_probability_before": stage3_probability_before,
                "stage3_prediction_before": stage3_prediction_before,
                "stage3_probability_after": stage3_probability_after,
                "stage3_prediction_after": stage3_prediction_after,
                "stage3_prediction_changed": int(stage3_prediction_before != stage3_prediction_after),
            }
        )

    output_dataframe = pd.DataFrame(output_rows)
    output_dataframe.to_csv(output_csv_path, index=False, encoding="utf-8")

    stage1_flip_rate = float(output_dataframe["stage1_prediction_changed"].mean())
    stage3_flip_rate = float(output_dataframe["stage3_prediction_changed"].mean())

    print("Robustness mini-benchmark complete.")
    print("Number of samples:", len(output_dataframe))
    print("Stage 1 prediction flip rate:", round(stage1_flip_rate, 3))
    print("Stage 3 prediction flip rate:", round(stage3_flip_rate, 3))
    print("Saved results to:", output_csv_path)


if __name__ == "__main__":
    main()