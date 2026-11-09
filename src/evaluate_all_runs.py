
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

STRATEGIES = {
    "zero_shot_v2": {
        1: "test_zero_shot_v2_results.jsonl",
        2: "test_zero_shot_v2_run2_results.jsonl",
        3: "test_zero_shot_v2_run3_results.jsonl",
    },
    "few_shot_k3": {
        1: "test_few_shot_k3_results.jsonl",
        2: "test_few_shot_k3_run2_results.jsonl",
        3: "test_few_shot_k3_run3_results.jsonl",
    },
    "few_shot_k5": {
        1: "test_few_shot_k5_results.jsonl",
        2: "test_few_shot_k5_run2_results.jsonl",
        3: "test_few_shot_k5_run3_results.jsonl",
    },
    "few_shot_k8": {
        1: "test_few_shot_k8_results.jsonl",
        2: "test_few_shot_k8_run2_results.jsonl",
        3: "test_few_shot_k8_run3_results.jsonl",
    },
    "cot_revised": {
        1: "test_cot_controlled_results.jsonl",
        2: "test_cot_controlled_run2_results.jsonl",
        3: "test_cot_controlled_run3_results.jsonl",
    },
    "cot_original": {
        1: "test_cot_results.jsonl",
    },
}

CLAIM_LABELS = [
    "own_damage", "third_party_liability", "theft",
    "total_loss", "glass_only", "not_a_claim"
]

URGENCY_LABELS = ["high", "medium", "low"]

EXTRACTION_FIELDS = [
    "policy_number", "incident_date", "location",
    "vehicle_reg_no", "estimated_amount_inr",
    "injury_reported", "police_report_filed"
]


def load_gold():
    path = DATA_DIR / "test.jsonl"
    records = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                records[obj["id"]] = {
                    "message": obj.get("message", ""),
                    "label": obj["label"]
                }

    return records


def load_run(filename):
    path = OUTPUTS_DIR / filename

    if not path.exists():
        print(f"WARNING: Missing file: {filename}")
        return {}

    results = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            if obj.get("status") != "success":
                continue

            if obj.get("dataset") != "test.jsonl":
                continue

            record_id = obj.get("record_id")

            if record_id:
                results[record_id] = obj

    return results


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


def get_prediction(result):
    prediction = result.get("parsed_output")
    return prediction if isinstance(prediction, dict) else {}


def evaluate_run(strategy, run_number, filename, gold):
    predictions = load_run(filename)

    rows = []

    for record_id, record in gold.items():
        result = predictions.get(record_id)

        if result is None:
            continue

        prediction = get_prediction(result)

        rows.append({
            "record_id": record_id,
            "message": record["message"],
            "gold": record["label"],
            "prediction": prediction,
            "result": result
        })

    total = len(rows)

    summary = {
        "strategy": strategy,
        "run": run_number,
        "prediction_file": filename,
        "records_expected": len(gold),
        "records_evaluated": total,
        "missing_records": len(gold) - total,
        "json_validity": (
            sum(bool(r["result"].get("json_valid")) for r in rows) / total
            if total else 0
        )
    }

    # Classification metrics

    for field, labels in [
        ("claim_type", CLAIM_LABELS),
        ("urgency", URGENCY_LABELS)
    ]:

        y_true = [r["gold"].get(field) for r in rows]

        y_pred = [
            r["prediction"].get(field)
            if r["prediction"].get(field) in labels
            else "__invalid__"
            for r in rows
        ]

        summary[f"{field}_accuracy"] = (
            accuracy_score(y_true, y_pred) if rows else 0
        )

        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0
        )

        summary[f"{field}_macro_f1"] = f1

    # Extraction metrics

    tp = fp = fn = exact = total_fields = 0

    for field in EXTRACTION_FIELDS:

        for row in rows:
            gold_value = row["gold"].get(field)
            predicted_value = row["prediction"].get(field)

            total_fields += 1

            if gold_value == predicted_value:
                exact += 1

            if gold_value is not None and predicted_value is not None:

                if normalize(gold_value) == normalize(predicted_value):
                    tp += 1
                else:
                    fp += 1
                    fn += 1

            elif predicted_value is not None:
                fp += 1

            elif gold_value is not None:
                fn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    summary["extraction_exact_match"] = (
        exact / total_fields if total_fields else 0
    )
    summary["extraction_precision"] = precision
    summary["extraction_recall"] = recall
    summary["extraction_f1"] = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    # Injury recall

    injury_cases = [
        r for r in rows
        if r["gold"].get("injury_reported") is True
    ]

    summary["injury_cases"] = len(injury_cases)

    summary["injury_recall"] = (
        sum(
            r["prediction"].get("injury_reported") is True
            for r in injury_cases
        ) / len(injury_cases)
        if injury_cases else 0
    )

    # High urgency recall

    high_cases = [
        r for r in rows
        if r["gold"].get("urgency") == "high"
    ]

    summary["high_urgency_cases"] = len(high_cases)

    summary["high_urgency_recall"] = (
        sum(
            r["prediction"].get("urgency") == "high"
            for r in high_cases
        ) / len(high_cases)
        if high_cases else 0
    )

    # Latency and token usage

    latencies = [
        r["result"].get("latency_seconds")
        for r in rows
        if isinstance(r["result"].get("latency_seconds"), (int, float))
    ]

    input_tokens = [
        r["result"].get("input_tokens")
        for r in rows
        if isinstance(r["result"].get("input_tokens"), (int, float))
    ]

    output_tokens = [
        r["result"].get("output_tokens")
        for r in rows
        if isinstance(r["result"].get("output_tokens"), (int, float))
    ]

    summary["average_latency_seconds"] = (
        sum(latencies) / len(latencies) if latencies else 0
    )

    summary["average_input_tokens"] = (
        sum(input_tokens) / len(input_tokens) if input_tokens else 0
    )

    summary["average_output_tokens"] = (
        sum(output_tokens) / len(output_tokens) if output_tokens else 0
    )

    summary["average_total_tokens"] = (
        summary["average_input_tokens"]
        + summary["average_output_tokens"]
    )

    # Lexical hallucination heuristic

    predicted_values = 0
    unsupported_values = 0

    for row in rows:

        message = normalize(row["message"]) or ""

        for field in EXTRACTION_FIELDS:

            value = row["prediction"].get(field)

            if value is None:
                continue

            predicted_values += 1

            if field in [
                "incident_date",
                "injury_reported",
                "police_report_filed"
            ]:
                continue

            value_normalized = normalize(value) or ""

            supported = value_normalized in message

            if field == "estimated_amount_inr":
                digits = re.sub(r"\D", "", str(value))
                message_digits = re.sub(r"\D", "", row["message"])

                supported = supported or (
                    bool(digits) and digits in message_digits
                )

            if not supported:
                unsupported_values += 1

    summary["predicted_extraction_values"] = predicted_values
    summary["unsupported_values_heuristic"] = unsupported_values

    summary["hallucination_rate_heuristic"] = (
        unsupported_values / predicted_values
        if predicted_values else 0
    )

    return summary, predictions


def calculate_agreement(strategy, runs, gold):
    run_numbers = sorted(runs)

    if len(run_numbers) < 2:
        return {
            "strategy": strategy,
            "runs_available": len(run_numbers),
            "records_compared": 0,
            "classification_agreement": None,
            "full_output_agreement": None,
            "note": "Only one run available."
        }

    common_ids = set(gold)

    for run_number in run_numbers:
        common_ids &= set(runs[run_number])

    class_matches = 0
    class_total = 0
    full_matches = 0

    for record_id in common_ids:

        outputs = [
            get_prediction(runs[n][record_id])
            for n in run_numbers
        ]

        for field in ["claim_type", "urgency"]:

            values = [p.get(field) for p in outputs]

            class_matches += len(set(map(str, values))) == 1
            class_total += 1

        serialized = [
            json.dumps(p, sort_keys=True)
            for p in outputs
        ]

        full_matches += len(set(serialized)) == 1

    return {
        "strategy": strategy,
        "runs_available": len(run_numbers),
        "records_compared": len(common_ids),
        "classification_agreement": (
            class_matches / class_total if class_total else 0
        ),
        "full_output_agreement": (
            full_matches / len(common_ids) if common_ids else 0
        ),
        "note": "Exact agreement across all available runs."
    }


def main():

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    gold = load_gold()

    print(f"Loaded {len(gold)} TEST records.")

    all_summaries = []
    agreement_results = []

    for strategy, files in STRATEGIES.items():

        print(f"\nEvaluating {strategy}...")

        loaded_runs = {}

        for run_number, filename in files.items():

            summary, predictions = evaluate_run(
                strategy,
                run_number,
                filename,
                gold
            )

            all_summaries.append(summary)
            loaded_runs[run_number] = predictions

            print(
                f"Run {run_number}: "
                f"{summary['records_evaluated']}/"
                f"{summary['records_expected']} records | "
                f"Claim accuracy: "
                f"{summary['claim_type_accuracy']:.4f} | "
                f"Urgency accuracy: "
                f"{summary['urgency_accuracy']:.4f}"
            )

        agreement_results.append(
            calculate_agreement(strategy, loaded_runs, gold)
        )

    run_df = pd.DataFrame(all_summaries)

    agreement_df = pd.DataFrame(agreement_results)

    # Calculate mean and standard deviation across runs

    excluded = {
        "strategy",
        "run",
        "prediction_file",
        "records_expected",
        "records_evaluated",
        "missing_records",
        "injury_cases",
        "high_urgency_cases"
    }

    metric_columns = [
        col for col in run_df.columns
        if col not in excluded
        and pd.api.types.is_numeric_dtype(run_df[col])
    ]

    aggregate_rows = []

    for strategy, group in run_df.groupby("strategy", sort=False):

        result = {
            "strategy": strategy,
            "runs_included": len(group)
        }

        for col in metric_columns:

            result[f"{col}_mean"] = group[col].mean()

            result[f"{col}_std"] = (
                group[col].std(ddof=1)
                if len(group) > 1 else 0
            )

        aggregate_rows.append(result)

    aggregate_df = pd.DataFrame(aggregate_rows)

    # Save all three CSV files

    run_df.to_csv(
        OUTPUTS_DIR / "test_repeated_run_metrics.csv",
        index=False
    )

    aggregate_df.to_csv(
        OUTPUTS_DIR / "test_repeated_run_summary.csv",
        index=False
    )

    agreement_df.to_csv(
        OUTPUTS_DIR / "test_run_to_run_agreement.csv",
        index=False
    )

    print("\n======================================")
    print("REPEATED-RUN EVALUATION COMPLETED")
    print("======================================")

    print("\nGenerated files:")

    print("- test_repeated_run_metrics.csv")
    print("- test_repeated_run_summary.csv")
    print("- test_run_to_run_agreement.csv")

    print("\nImportant notes:")
    print("- Original CoT has only one run.")
    print("- Hallucination rate is a lexical heuristic.")
    print("- Missing predictions are reported, not silently ignored.")


if __name__ == "__main__":
    main()
