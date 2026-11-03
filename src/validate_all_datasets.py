import json
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

DATASETS = {
    "dev": {
        "path": DATA_DIR / "dev.jsonl",
        "expected": 20
    },
    "test": {
        "path": DATA_DIR / "test.jsonl",
        "expected": 60
    }
}

VALID_CLAIM_TYPES = {
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim"
}

VALID_URGENCY = {"high", "medium", "low"}
REQUIRED_FIELDS = {
    "claim_type",
    "urgency",
    "policy_number",
    "incident_date",
    "location",
    "vehicle_reg_no",
    "estimated_amount_inr",
    "injury_reported",
    "police_report_filed"
}

BOOLEAN_FIELDS = {
    "injury_reported",
    "police_report_filed"
}

all_ids = set()
claim_distribution = Counter()
urgency_distribution = Counter()
total_records = 0
errors = []


def validate_file(name, config):

    global total_records

    records = []

    with open(config["path"], "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"{name}: Invalid JSON at line {line_number}")
                continue

            records.append(record)

            record_id = record.get("id")

            if not record_id:
                errors.append(f"{name}: Missing ID at line {line_number}")

            elif record_id in all_ids:
                errors.append(f"{name}: Duplicate ID {record_id}")

            else:
                all_ids.add(record_id)

            label = record.get("label", {})
                        # Check all nine required fields
            if not isinstance(label, dict) or set(label.keys()) != REQUIRED_FIELDS:
                errors.append(
                    f"{name}: Incorrect label fields at line {line_number}"
                )
                continue

            # Check policy number and location
            for field in ["policy_number", "location"]:
                if label[field] is not None and not isinstance(label[field], str):
                    errors.append(
                        f"{name}: Invalid {field} at line {line_number}"
                    )

            # Check incident date
            incident_date = label["incident_date"]

            if incident_date is not None:
                from datetime import date

                try:
                    date.fromisoformat(incident_date)
                except (ValueError, TypeError):
                    errors.append(
                        f"{name}: Invalid date at line {line_number}"
                    )

            # Check vehicle registration
            reg_no = label["vehicle_reg_no"]

            if reg_no is not None:
                if (
                    not isinstance(reg_no, str)
                    or reg_no != reg_no.upper()
                    or " " in reg_no
                ):
                    errors.append(
                        f"{name}: Invalid registration at line {line_number}"
                    )

            # Check estimated amount
            amount = label["estimated_amount_inr"]

            if amount is not None:
                if type(amount) is not int or amount < 0:
                    errors.append(
                        f"{name}: Invalid amount at line {line_number}"
                    )

            # Check boolean fields
            for field in BOOLEAN_FIELDS:
                if label[field] is not None:
                    if not isinstance(label[field], bool):
                        errors.append(
                            f"{name}: Invalid {field} at line {line_number}"
                        )

            if label.get("claim_type") not in VALID_CLAIM_TYPES:
                errors.append(f"{name}: Invalid claim type at line {line_number}")
            else:
                claim_distribution[label["claim_type"]] += 1

            if label.get("urgency") not in VALID_URGENCY:
                errors.append(f"{name}: Invalid urgency at line {line_number}")
            else:
                urgency_distribution[label["urgency"]] += 1

    total_records += len(records)

    if len(records) != config["expected"]:
        errors.append(
            f"{name}: Expected {config['expected']} records, found {len(records)}"
        )

    print(f"{name.upper()} records: {len(records)}")


for name, config in DATASETS.items():
    validate_file(name, config)

print("\nCLAIMFAST DATASET REPORT")
print("-----------------------")
print(f"Total records: {total_records}")
print(f"Unique IDs: {len(all_ids)}")

print("\nClaim type distribution:")
for claim_type, count in sorted(claim_distribution.items()):
    print(f"{claim_type}: {count}")

print("\nUrgency distribution:")
for urgency, count in sorted(urgency_distribution.items()):
    print(f"{urgency}: {count}")

print("\nValidation errors:")

if errors:
    for error in errors:
        print("-", error)
else:
    print("No errors found in the checks performed!")