
import csv
import json
from pathlib import Path
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

# ==================================================
# 1. PROJECT PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

PROMPTS = [
    "zero_shot",
    "few_shot_k3",
    "few_shot_k5",
    "few_shot_k8",
    "cot"
]

FIELDS = [
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed"
]

CLAIM_TYPES = [
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim"
]

URGENCY_LEVELS = ["high", "medium", "low"]


# ==================================================
# 2. LOAD DEVELOPMENT DATA
# ==================================================

def load_dev_data():
    """Load labelled development examples."""

    dev_file = DATA_DIR / "dev.jsonl"
    records = []

    with open(dev_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    dev_data = {}

    for record in records:

        record_id = record.get("id", record.get("record_id"))

        if record_id is None:
            raise ValueError(
                "A development record is missing its ID."
            )

        if "label" not in record:
            raise ValueError(
                f"Record {record_id} is missing its label object."
            )

        dev_data[record_id] = record["label"]

    return dev_data


# ==================================================
# 3. LOAD MODEL RESULTS
# ==================================================

def load_model_results(prompt_name):
    """Load saved predictions for one prompt strategy."""

    results_file = OUTPUTS_DIR / f"{prompt_name}_results.jsonl"

    if not results_file.exists():
        print(f"Warning: {results_file.name} not found.")
        return []

    results = []

    with open(results_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))

    return results


# ==================================================
# 4. VALIDATE PREDICTED JSON SCHEMA
# ==================================================

def validate_schema(prediction):
    """Check whether a prediction follows the 9-field schema."""

    if not isinstance(prediction, dict):
        return False

    if set(prediction.keys()) != set(FIELDS):
        return False

    if prediction["claim_type"] not in CLAIM_TYPES:
        return False

    if prediction["urgency"] not in URGENCY_LEVELS:
        return False

    string_fields = [
        "policy_number",
        "incident_date",
        "location",
        "vehicle_reg_no"
    ]

    for field in string_fields:
        value = prediction[field]

        if value is not None and not isinstance(value, str):
            return False

    amount = prediction["estimated_amount_inr"]

    if amount is not None and (
        not isinstance(amount, int) or isinstance(amount, bool)
    ):
        return False

    for field in ["injury_reported", "police_report_filed"]:
        value = prediction[field]

        if value is not None and not isinstance(value, bool):
            return False

    return True


# ==================================================
# 5. CLASSIFICATION METRICS
# ==================================================

def calculate_classification_metrics(actual, predicted, labels):
    """Calculate accuracy, macro precision, recall and F1."""

    accuracy = accuracy_score(actual, predicted)

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(precision, 4),
        "macro_recall": round(recall, 4),
        "macro_f1": round(f1, 4)
    }


# ==================================================
# 6. SAVE CONFUSION MATRIX
# ==================================================

def save_confusion_matrix(actual, predicted, labels, filename):
    """Save confusion matrix as CSV."""

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    output_file = OUTPUTS_DIR / filename

    with open(output_file, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow(["Actual / Predicted"] + labels)

        for label, row in zip(labels, matrix):
            writer.writerow([label] + row.tolist())


# ==================================================
# 7. EVALUATE ONE PROMPT STRATEGY
# ==================================================

def evaluate_prompt(prompt_name, dev_data):

    print(f"\nEvaluating: {prompt_name}")

    results = load_model_results(prompt_name)

    # Remove duplicate entries using record ID and run number.
    # If duplicates exist, retain the latest timestamp.

    unique_results = {}

    for result in results:

        record_id = result.get(
            "record_id",
            result.get("id")
        )

        if record_id is None:
            continue

        key = (
            record_id,
            result.get("run_number", 1)
        )

        timestamp = result.get("timestamp", "")

        if (
            key not in unique_results
            or timestamp >= unique_results[key].get("timestamp", "")
        ):
            unique_results[key] = result

    results = list(unique_results.values())

    actual_claim = []
    predicted_claim = []

    actual_urgency = []
    predicted_urgency = []

    field_correct = Counter()
    field_total = Counter()

    detailed_rows = []

    json_valid_count = 0
    schema_valid_count = 0

    total_input_tokens = 0
    total_output_tokens = 0
    total_latency = 0

    matched_records = 0

    for result in results:

        record_id = result.get(
            "record_id",
            result.get("id")
        )

        if record_id not in dev_data:
            continue

        actual = dev_data[record_id]
        prediction = result.get("parsed_output")

        # Handle predictions stored as JSON strings.
        if isinstance(prediction, str):
            try:
                prediction = json.loads(prediction)
            except json.JSONDecodeError:
                prediction = None

        matched_records += 1

        is_json_valid = result.get("json_valid") is True
        is_schema_valid = validate_schema(prediction)

        if is_json_valid:
            json_valid_count += 1

        if is_schema_valid:
            schema_valid_count += 1

        # Classification metrics use schema-valid predictions only.
        if is_schema_valid:

            actual_claim.append(actual["claim_type"])
            predicted_claim.append(prediction["claim_type"])

            actual_urgency.append(actual["urgency"])
            predicted_urgency.append(prediction["urgency"])

        # Compare all nine fields.
        for field in FIELDS:

            actual_value = actual.get(field)

            predicted_value = (
                prediction.get(field)
                if isinstance(prediction, dict)
                else None
            )

            field_total[field] += 1

            correct = predicted_value == actual_value

            if correct:
                field_correct[field] += 1

            detailed_rows.append({
                "prompt": prompt_name,
                "id": record_id,
                "field": field,
                "actual": actual_value,
                "predicted": predicted_value,
                "correct": correct,
                "json_valid": is_json_valid,
                "schema_valid": is_schema_valid
            })

        total_input_tokens += result.get("input_tokens") or 0
        total_output_tokens += result.get("output_tokens") or 0
        total_latency += result.get("latency_seconds") or 0

    # Classification metrics.
    claim_metrics = (
        calculate_classification_metrics(
            actual_claim,
            predicted_claim,
            CLAIM_TYPES
        )
        if actual_claim else {}
    )

    urgency_metrics = (
        calculate_classification_metrics(
            actual_urgency,
            predicted_urgency,
            URGENCY_LEVELS
        )
        if actual_urgency else {}
    )

    # Save confusion matrices.
    if actual_claim:
        save_confusion_matrix(
            actual_claim,
            predicted_claim,
            CLAIM_TYPES,
            f"confusion_claim_type_{prompt_name}.csv"
        )

    if actual_urgency:
        save_confusion_matrix(
            actual_urgency,
            predicted_urgency,
            URGENCY_LEVELS,
            f"confusion_urgency_{prompt_name}.csv"
        )

    average_latency = (
        total_latency / matched_records
        if matched_records else 0
    )

    # Build summary.
    summary = {
        "prompt": prompt_name,
        "records_evaluated": matched_records,

        "json_validity_rate": round(
            json_valid_count / matched_records, 4
        ) if matched_records else 0,

        "schema_validity_rate": round(
            schema_valid_count / matched_records, 4
        ) if matched_records else 0,

        "classification_records": len(actual_claim),

        "claim_type_accuracy": claim_metrics.get("accuracy"),
        "claim_type_macro_f1": claim_metrics.get("macro_f1"),

        "urgency_accuracy": urgency_metrics.get("accuracy"),
        "urgency_macro_f1": urgency_metrics.get("macro_f1"),

        "average_input_tokens": round(
            total_input_tokens / matched_records, 2
        ) if matched_records else 0,

        "average_output_tokens": round(
            total_output_tokens / matched_records, 2
        ) if matched_records else 0,

        "average_latency_seconds": round(average_latency, 2)
    }

    # Field-level exact match.
    for field in FIELDS:

        summary[f"{field}_exact_match"] = round(
            field_correct[field] / field_total[field], 4
        ) if field_total[field] else 0

    # Display results.
    print(f"Records evaluated: {matched_records}")
    print(f"Classification records: {len(actual_claim)}")
    print(f"JSON validity: {summary['json_validity_rate']}")
    print(f"Schema validity: {summary['schema_validity_rate']}")
    print(f"Claim type accuracy: {summary['claim_type_accuracy']}")
    print(f"Claim type macro-F1: {summary['claim_type_macro_f1']}")
    print(f"Urgency accuracy: {summary['urgency_accuracy']}")
    print(f"Urgency macro-F1: {summary['urgency_macro_f1']}")
    print(f"Average latency: {average_latency:.2f} seconds")

    return summary, detailed_rows


# ==================================================
# 8. RUN ALL EVALUATIONS
# ==================================================

def main():

    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("=" * 55)
    print("CLAIMFAST - PROMPT EVALUATION")
    print("=" * 55)

    dev_data = load_dev_data()

    all_summaries = []
    all_details = []

    for prompt_name in PROMPTS:

        summary, details = evaluate_prompt(
            prompt_name,
            dev_data
        )

        all_summaries.append(summary)
        all_details.extend(details)

    # Save evaluation summary.
    if all_summaries:

        summary_file = OUTPUTS_DIR / "evaluation_summary.csv"

        with open(
            summary_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=all_summaries[0].keys()
            )

            writer.writeheader()
            writer.writerows(all_summaries)

    # Save detailed field-level results.
    if all_details:

        details_file = OUTPUTS_DIR / "detailed_evaluation.csv"

        with open(
            details_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=all_details[0].keys()
            )

            writer.writeheader()
            writer.writerows(all_details)

    print("\n" + "=" * 55)
    print("EVALUATION COMPLETED")
    print("=" * 55)

    print("\nFiles saved in outputs/:")
    print("- evaluation_summary.csv")
    print("- detailed_evaluation.csv")
    print("- confusion matrix CSV files")

    print(
        "\nNote: These are preliminary development results. "
        "Only saved predictions are evaluated. "
        "No new API calls are made."
    )


if __name__ == "__main__":
    main()
