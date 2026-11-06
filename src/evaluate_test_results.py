
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

PROMPT_NAMES = [
    "zero_shot",
    "few_shot_k3",
    "few_shot_k5",
    "few_shot_k8",
    "cot"
]

CLASSIFICATION_FIELDS = [
    "claim_type",
    "urgency"
]

EXTRACTION_FIELDS = [
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed"
]

REQUIRED_FIELDS = CLASSIFICATION_FIELDS + EXTRACTION_FIELDS

VALID_CLAIM_TYPES = [
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim"
]

VALID_URGENCIES = [
    "high",
    "medium",
    "low"
]


# ==========================================
# 2. LOAD TEST DATASET
# ==========================================

def load_test_data():

    dataset_path = DATA_DIR / "test.jsonl"

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    records = {}

    with dataset_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            records[record["id"]] = {
                "message": record["message"],
                "label": record["label"]
            }

    print(f"Gold test records loaded: {len(records)}")

    return records


# ==========================================
# 3. LOAD TEST PREDICTIONS
# ==========================================

def load_predictions(prompt_name):

    output_path = OUTPUTS_DIR / (
        f"test_{prompt_name}_results.jsonl"
    )

    if not output_path.exists():
        return {}

    predictions = {}

    with output_path.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            try:
                result = json.loads(line)
            except json.JSONDecodeError:
                continue

            if result.get("status") != "success":
                continue

            if result.get("dataset") != "test.jsonl":
                continue

            if result.get("run_number") != 1:
                continue

            record_id = result.get("record_id")

            if not record_id:
                continue

            timestamp = result.get("timestamp", "")

            old_result = predictions.get(record_id)

            if old_result is None or timestamp >= old_result.get(
                "timestamp", ""
            ):
                predictions[record_id] = result

    return predictions


# ==========================================
# 4. SCHEMA VALIDATION
# ==========================================

def validate_schema(prediction):

    if not isinstance(prediction, dict):
        return False

    if set(prediction.keys()) != set(REQUIRED_FIELDS):
        return False

    if prediction["claim_type"] not in VALID_CLAIM_TYPES:
        return False

    if prediction["urgency"] not in VALID_URGENCIES:
        return False

    for field in EXTRACTION_FIELDS:

        value = prediction[field]

        if value is None:
            continue

        if field in [
            "policy_number",
            "incident_date",
            "location",
            "vehicle_reg_no"
        ]:
            if not isinstance(value, str):
                return False

        elif field == "estimated_amount_inr":

            if isinstance(value, bool) or not isinstance(value, int):
                return False

        elif field in [
            "injury_reported",
            "police_report_filed"
        ]:
            if not isinstance(value, bool):
                return False

    if prediction["vehicle_reg_no"] is not None:

        registration = prediction["vehicle_reg_no"]

        if registration != registration.upper():
            return False

        if " " in registration:
            return False

    if prediction["incident_date"] is not None:

        try:
            from datetime import datetime

            datetime.strptime(
                prediction["incident_date"],
                "%Y-%m-%d"
            )

        except ValueError:
            return False

    return True


# ==========================================
# 5. NORMALISE VALUES
# ==========================================

def normalize(value):

    if value is None:
        return None

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (int, float)):
        return str(value)

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower().strip()
    )


# ==========================================
# 6. EXTRACTION METRICS
# ==========================================

def calculate_extraction_metrics(gold_values, predicted_values):

    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_matches = 0

    total = len(gold_values)

    for gold, predicted in zip(
        gold_values,
        predicted_values
    ):

        if gold == predicted:
            exact_matches += 1

        gold_present = gold is not None
        predicted_present = predicted is not None

        if gold_present and predicted_present:

            if normalize(gold) == normalize(predicted):
                true_positive += 1
            else:
                false_positive += 1
                false_negative += 1

        elif predicted_present:
            false_positive += 1

        elif gold_present:
            false_negative += 1

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    exact_match = (
        exact_matches / total if total else 0
    )

    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================
# 7. HALLUCINATION HEURISTIC
# ==========================================

def value_supported_by_message(field, value, message):

    if value is None:
        return True

    # Boolean values require semantic interpretation.
    # Dates may be normalised from written date formats.
    if field in [
        "incident_date",
        "injury_reported",
        "police_report_filed"
    ]:
        return True

    message_normalized = normalize(message)
    value_normalized = normalize(value)

    if not value_normalized:
        return True

    if value_normalized in message_normalized:
        return True

    # Check numeric amounts without currency punctuation.
    if field == "estimated_amount_inr":

        digits = re.sub(r"\D", "", str(value))
        message_digits = re.sub(r"\D", "", message)

        if digits and digits in message_digits:
            return True

    return False


def calculate_hallucination_metrics(rows):

    total_values = 0
    unsupported_values = 0

    for row in rows:

        prediction = row["prediction"]

        if not isinstance(prediction, dict):
            continue

        for field in EXTRACTION_FIELDS:

            value = prediction.get(field)

            if value is None:
                continue

            total_values += 1

            if not value_supported_by_message(
                field,
                value,
                row["message"]
            ):
                unsupported_values += 1

    return {
        "predicted_extraction_values": total_values,
        "unsupported_values_heuristic": unsupported_values,
        "hallucination_rate_heuristic": (
            unsupported_values / total_values
            if total_values else 0
        )
    }


# ==========================================
# 8. EVALUATE ONE PROMPT STRATEGY
# ==========================================

def evaluate_strategy(prompt_name, test_data):

    predictions = load_predictions(prompt_name)

    rows = []

    for record_id, record in test_data.items():

        result = predictions.get(record_id)

        if result is None:
            continue

        prediction = result.get("parsed_output")

        if not isinstance(prediction, dict):
            prediction = {}

        gold = record["label"]

        rows.append({
            "record_id": record_id,
            "message": record["message"],
            "gold": gold,
            "prediction": prediction,
            "json_valid": result.get("json_valid", False),
            "schema_valid": validate_schema(prediction),
            "latency": result.get("latency_seconds"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens")
        })

    summary = {
        "prompt_strategy": prompt_name,
        "records_expected": len(test_data),
        "records_evaluated": len(rows),
        "missing_records": len(test_data) - len(rows),
        "json_validity": (
            sum(row["json_valid"] for row in rows) / len(rows)
            if rows else 0
        ),
        "schema_validity": (
            sum(row["schema_valid"] for row in rows) / len(rows)
            if rows else 0
        )
    }

    # --------------------------------------
    # CLASSIFICATION METRICS
    # --------------------------------------

    for field, valid_labels in [
        ("claim_type", VALID_CLAIM_TYPES),
        ("urgency", VALID_URGENCIES)
    ]:

        y_true = [
            row["gold"][field]
            for row in rows
        ]

        y_pred = [
            row["prediction"].get(field)
            for row in rows
        ]

        if not rows:
            continue

        safe_predictions = [
            value if value in valid_labels else "__invalid__"
            for value in y_pred
        ]

        summary[f"{field}_accuracy"] = accuracy_score(
            y_true,
            y_pred
        )

        precision, recall, f1, _ = (
            precision_recall_fscore_support(
                y_true,
                safe_predictions,
                labels=valid_labels,
                average="macro",
                zero_division=0
            )
        )

        summary[f"{field}_macro_precision"] = precision
        summary[f"{field}_macro_recall"] = recall
        summary[f"{field}_macro_f1"] = f1

        matrix = confusion_matrix(
            y_true,
            safe_predictions,
            labels=valid_labels
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=valid_labels,
            columns=valid_labels
        )

        matrix_df.index.name = "Actual"
        matrix_df.columns.name = "Predicted"

        matrix_df.to_csv(
            OUTPUTS_DIR /
            f"test_{prompt_name}_{field}_confusion_matrix.csv"
        )

    # --------------------------------------
    # EXTRACTION METRICS
    # --------------------------------------

    all_gold = []
    all_predicted = []

    for field in EXTRACTION_FIELDS:

        gold_values = [
            row["gold"][field]
            for row in rows
        ]

        predicted_values = [
            row["prediction"].get(field)
            for row in rows
        ]

        metrics = calculate_extraction_metrics(
            gold_values,
            predicted_values
        )

        for metric_name, value in metrics.items():

            summary[f"{field}_{metric_name}"] = value

        all_gold.extend(gold_values)
        all_predicted.extend(predicted_values)

    overall = calculate_extraction_metrics(
        all_gold,
        all_predicted
    )

    summary["overall_extraction_exact_match"] = overall["exact_match"]
    summary["overall_extraction_precision"] = overall["precision"]
    summary["overall_extraction_recall"] = overall["recall"]
    summary["overall_extraction_f1"] = overall["f1"]

    # --------------------------------------
    # INJURY RECALL
    # --------------------------------------

    injury_cases = [
        row for row in rows
        if row["gold"]["injury_reported"] is True
    ]

    injury_detected = sum(
        row["prediction"].get("injury_reported") is True
        for row in injury_cases
    )

    summary["injury_cases"] = len(injury_cases)

    summary["injury_recall"] = (
        injury_detected / len(injury_cases)
        if injury_cases else 0
    )

    # --------------------------------------
    # HIGH URGENCY RECALL
    # --------------------------------------

    high_cases = [
        row for row in rows
        if row["gold"]["urgency"] == "high"
    ]

    high_detected = sum(
        row["prediction"].get("urgency") == "high"
        for row in high_cases
    )

    summary["high_urgency_cases"] = len(high_cases)

    summary["high_urgency_recall"] = (
        high_detected / len(high_cases)
        if high_cases else 0
    )

    # --------------------------------------
    # LATENCY AND TOKEN USAGE
    # --------------------------------------

    latencies = [
        row["latency"]
        for row in rows
        if isinstance(row["latency"], (int, float))
    ]

    input_tokens = [
        row["input_tokens"]
        for row in rows
        if isinstance(row["input_tokens"], (int, float))
    ]

    output_tokens = [
        row["output_tokens"]
        for row in rows
        if isinstance(row["output_tokens"], (int, float))
    ]

    summary["average_latency_seconds"] = (
        sum(latencies) / len(latencies)
        if latencies else 0
    )

    summary["average_input_tokens"] = (
        sum(input_tokens) / len(input_tokens)
        if input_tokens else 0
    )

    summary["average_output_tokens"] = (
        sum(output_tokens) / len(output_tokens)
        if output_tokens else 0
    )

    summary["total_tokens"] = (
        sum(input_tokens) + sum(output_tokens)
    )

    # --------------------------------------
    # HALLUCINATION HEURISTIC
    # --------------------------------------

    summary.update(
        calculate_hallucination_metrics(rows)
    )

    return summary, rows


# ==========================================
# 9. SAVE DETAILED EVALUATION
# ==========================================

def create_detailed_rows(prompt_name, rows):

    details = []

    for row in rows:

        detail = {
            "prompt_strategy": prompt_name,
            "record_id": row["record_id"],
            "json_valid": row["json_valid"],
            "schema_valid": row["schema_valid"],
            "latency_seconds": row["latency"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"]
        }

        for field in REQUIRED_FIELDS:

            gold = row["gold"][field]
            predicted = row["prediction"].get(field)

            detail[f"gold_{field}"] = gold
            detail[f"predicted_{field}"] = predicted

            detail[f"{field}_correct"] = (
                gold == predicted
            )

        details.append(detail)

    return details


# ==========================================
# 10. RUN ALL EVALUATIONS
# ==========================================

def main():

    OUTPUTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    test_data = load_test_data()

    summaries = []
    all_details = []

    print("\n==========================================")
    print("       CLAIMFAST TEST EVALUATION")
    print("==========================================")

    for prompt_name in PROMPT_NAMES:

        print("\n" + "=" * 50)
        print(f"Evaluating: {prompt_name}")
        print("=" * 50)

        summary, rows = evaluate_strategy(
            prompt_name,
            test_data
        )

        summaries.append(summary)

        all_details.extend(
            create_detailed_rows(prompt_name, rows)
        )

        print(
            f"Records evaluated: "
            f"{summary['records_evaluated']}/"
            f"{summary['records_expected']}"
        )

        print(
            f"Missing records: {summary['missing_records']}"
        )

        print(
            f"JSON validity: {summary['json_validity']:.4f}"
        )

        print(
            f"Schema validity: {summary['schema_validity']:.4f}"
        )

        if rows:

            print(
                f"Claim type accuracy: "
                f"{summary.get('claim_type_accuracy', 0):.4f}"
            )

            print(
                f"Claim type macro-F1: "
                f"{summary.get('claim_type_macro_f1', 0):.4f}"
            )

            print(
                f"Urgency accuracy: "
                f"{summary.get('urgency_accuracy', 0):.4f}"
            )

            print(
                f"Urgency macro-F1: "
                f"{summary.get('urgency_macro_f1', 0):.4f}"
            )

            print(
                f"Extraction F1: "
                f"{summary['overall_extraction_f1']:.4f}"
            )

            print(
                f"Injury recall: "
                f"{summary['injury_recall']:.4f}"
            )

            print(
                f"High urgency recall: "
                f"{summary['high_urgency_recall']:.4f}"
            )

            print(
                f"Average latency: "
                f"{summary['average_latency_seconds']:.2f}s"
            )

    # Save summary
    pd.DataFrame(summaries).to_csv(
        OUTPUTS_DIR / "test_evaluation_summary.csv",
        index=False
    )

    # Save detailed evaluation
    pd.DataFrame(all_details).to_csv(
        OUTPUTS_DIR / "test_detailed_evaluation.csv",
        index=False
    )

    print("\n==========================================")
    print("       TEST EVALUATION COMPLETED")
    print("==========================================")

    print("Files saved in outputs/:")
    print("- test_evaluation_summary.csv")
    print("- test_detailed_evaluation.csv")
    print("- Test confusion matrix CSV files")

    print(
        "\nMissing predictions are excluded from scoring. "
        "Check missing_records before interpreting results."
    )

    print(
        "\nNote: Hallucination rate is a lexical heuristic. "
        "It requires manual review before being treated "
        "as a confirmed hallucination rate."
    )


# ==========================================
# 11. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()
