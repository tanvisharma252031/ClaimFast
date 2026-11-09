
import json
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

RUN_FILES = {
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

EXTRACTION_FIELDS = [
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed",
]


def normalize(value):
    if value is None:
        return None

    if isinstance(value, bool):
        return str(value).lower()

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(value).lower().strip()
    )


def load_gold():
    records = {}

    path = DATA_DIR / "test.jsonl"

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                records[item["id"]] = item

    return records


def load_predictions(filename):
    path = OUTPUTS_DIR / filename

    if not path.exists():
        print(f"WARNING: Missing file: {filename}")
        return {}

    predictions = {}

    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            if item.get("status") != "success":
                continue

            if item.get("dataset") != "test.jsonl":
                continue

            record_id = item.get("record_id")

            if record_id:
                predictions[record_id] = item

    return predictions


def add_error(
    errors,
    strategy,
    run,
    record_id,
    field,
    category,
    expected,
    predicted,
    severity,
    analysis
):
    errors.append({
        "prompt_strategy": strategy,
        "run": run,
        "record_id": record_id,
        "error_category": category,
        "field_name": field,
        "expected_value": expected,
        "predicted_value": predicted,
        "severity": severity,
        "analysis": analysis,
    })


def main():
    gold_records = load_gold()

    errors = []
    run_coverage = []

    for strategy, runs in RUN_FILES.items():

        print(f"\nAnalysing {strategy}...")

        for run_number, filename in runs.items():

            predictions = load_predictions(filename)

            run_coverage.append({
                "prompt_strategy": strategy,
                "run": run_number,
                "records_expected": len(gold_records),
                "records_found": len(predictions),
                "missing_records": (
                    len(gold_records) -
                    len(set(gold_records) & set(predictions))
                )
            })

            for record_id, gold_record in gold_records.items():

                result = predictions.get(record_id)

                if result is None:
                    add_error(
                        errors, strategy, run_number,
                        record_id, "entire_record",
                        "Missing prediction",
                        "Prediction expected",
                        None, "High",
                        "No successful prediction was found."
                    )
                    continue

                prediction = result.get("parsed_output")

                if not isinstance(prediction, dict):
                    add_error(
                        errors, strategy, run_number,
                        record_id, "JSON",
                        "Invalid or missing JSON",
                        "Valid JSON object",
                        prediction, "High",
                        "The model output could not be parsed as a JSON object."
                    )
                    continue

                gold = gold_record["label"]
                message = gold_record.get("message", "")

                # 1. Classification errors

                for field in ["claim_type", "urgency"]:

                    expected = gold.get(field)
                    predicted = prediction.get(field)

                    if expected != predicted:

                        severity = (
                            "Critical"
                            if field == "urgency"
                            and expected == "high"
                            else "High"
                        )

                        add_error(
                            errors, strategy, run_number,
                            record_id, field,
                            "Classification error"
                            if field == "claim_type"
                            else "Urgency error",
                            expected, predicted, severity,
                            f"Expected {expected}, received {predicted}."
                        )

                # 2. Extraction errors

                for field in EXTRACTION_FIELDS:

                    expected = gold.get(field)
                    predicted = prediction.get(field)

                    if expected == predicted:
                        continue

                    # Location differences are separately flagged.
                    if field == "location":

                        if (
                            expected is not None
                            and predicted is not None
                            and normalize(expected) == normalize(predicted)
                        ):
                            category = "Location formatting difference"
                            severity = "Low"
                            analysis = (
                                "Differences are limited to case or punctuation."
                            )

                        elif expected is not None and predicted is None:
                            category = "Location omission"
                            severity = "Medium"
                            analysis = (
                                "Gold label contains a location, "
                                "but prediction is null."
                            )

                        elif expected is None and predicted is not None:
                            category = "Potential unsupported extraction"
                            severity = "Medium"
                            analysis = (
                                "Prediction contains a location absent "
                                "from the gold label. Verify against message."
                            )

                        else:
                            category = "Location wording mismatch"
                            severity = "Low"
                            analysis = (
                                "Location wording differs. "
                                "Manual review is needed to determine "
                                "whether the meaning is equivalent."
                            )

                    elif expected is not None and predicted is None:
                        category = "Extraction omission"
                        severity = "Medium"
                        analysis = "A labelled value was not extracted."

                    elif expected is None and predicted is not None:
                        category = "Potential unsupported extraction"
                        severity = "Medium"
                        analysis = (
                            "The prediction contains a value where the "
                            "gold label is null. Verify the original message."
                        )

                    elif field == "injury_reported":
                        category = "Injury extraction error"
                        severity = "High"
                        analysis = (
                            "Injury information differs from the gold label."
                        )

                    else:
                        category = "Extraction mismatch"
                        severity = "Medium"
                        analysis = (
                            "The extracted value differs from the gold label."
                        )

                    add_error(
                        errors, strategy, run_number,
                        record_id, field, category,
                        expected, predicted, severity, analysis
                    )

    # Save detailed error analysis

    error_df = pd.DataFrame(errors)

    error_path = (
        OUTPUTS_DIR / "test_repeated_error_analysis.csv"
    )

    error_df.to_csv(error_path, index=False)

    # Build grouped error summary

    if not error_df.empty:

        summary = (
            error_df.groupby(
                ["prompt_strategy", "error_category"]
            )
            .agg(
                error_instances=("record_id", "count"),
                unique_test_records=("record_id", "nunique"),
                affected_runs=("run", "nunique")
            )
            .reset_index()
        )

    else:
        summary = pd.DataFrame(columns=[
            "prompt_strategy",
            "error_category",
            "error_instances",
            "unique_test_records",
            "affected_runs"
        ])

    summary_path = (
        OUTPUTS_DIR / "test_repeated_error_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    coverage_path = (
        OUTPUTS_DIR / "test_error_analysis_coverage.csv"
    )

    pd.DataFrame(run_coverage).to_csv(
        coverage_path, index=False
    )

    # Final terminal report

    print("\n======================================")
    print("REPEATED ERROR ANALYSIS COMPLETED")
    print("======================================")

    print(f"Total error instances: {len(errors)}")

    print("\nGenerated files:")

    print("- test_repeated_error_analysis.csv")
    print("- test_repeated_error_summary.csv")
    print("- test_error_analysis_coverage.csv")

    print("\nError categories:")

    if not summary.empty:
        print(summary.to_string(index=False))
    else:
        print("No errors found.")

    print("\nImportant:")
    print("- Error instances are not unique message counts.")
    print("- Location wording mismatches require manual review.")
    print("- Potential unsupported values are not confirmed hallucinations.")
    print("- Original CoT has only one available run.")


if __name__ == "__main__":
    main()
