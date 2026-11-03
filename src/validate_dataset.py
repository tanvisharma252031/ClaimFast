import json
from pathlib import Path

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset file
DATA_FILE = PROJECT_DIR / "data" / "dev.jsonl"

# Allowed classification values
VALID_CLAIM_TYPES = {
    "own_damage",
    "third_party_liability",
    "theft",
    "total_loss",
    "glass_only",
    "not_a_claim"
}

VALID_URGENCY = {"high", "medium", "low"}

# Required label fields
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

def validate_dataset():

    errors = []
    records = []
    seen_ids = set()

    # Read the JSONL file
    with open(DATA_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            try:
                record = json.loads(line)

            except json.JSONDecodeError:
                errors.append(
                    f"Line {line_number}: Invalid JSON"
                )
                continue

            records.append(record)

            # Check required top-level fields
            if not all(key in record for key in ["id", "message", "label"]):
                errors.append(
                    f"Line {line_number}: Missing top-level field"
                )
                continue

            # Check duplicate IDs
            record_id = record["id"]

            if record_id in seen_ids:
                errors.append(
                    f"Line {line_number}: Duplicate ID {record_id}"
                )

            seen_ids.add(record_id)

            # Check label fields
            label = record["label"]

            if set(label.keys()) != REQUIRED_FIELDS:
                errors.append(
                    f"Line {line_number}: Incorrect label fields"
                )
                continue

            # Check claim type
            if label["claim_type"] not in VALID_CLAIM_TYPES:
                errors.append(
                    f"Line {line_number}: Invalid claim type"
                )

            # Check urgency
            if label["urgency"] not in VALID_URGENCY:
                errors.append(
                    f"Line {line_number}: Invalid urgency"
                )

                        # Check policy number
            if label["policy_number"] is not None:
                if not isinstance(label["policy_number"], str):
                    errors.append(
                        f"Line {line_number}: Invalid policy number"
                    )

            # Check incident date
            incident_date = label["incident_date"]

            if incident_date is not None:
                try:
                    from datetime import date
                    date.fromisoformat(incident_date)

                except (ValueError, TypeError):
                    errors.append(
                        f"Line {line_number}: Invalid incident date"
                    )

            # Check location
            if label["location"] is not None:
                if not isinstance(label["location"], str):
                    errors.append(
                        f"Line {line_number}: Invalid location"
                    )

            # Check vehicle registration
            vehicle_reg = label["vehicle_reg_no"]

            if vehicle_reg is not None:
                if (
                    not isinstance(vehicle_reg, str)
                    or vehicle_reg != vehicle_reg.upper()
                    or " " in vehicle_reg
                ):
                    errors.append(
                        f"Line {line_number}: Invalid vehicle registration"
                    )

            # Check estimated amount
            amount = label["estimated_amount_inr"]

            if amount is not None:
                if type(amount) is not int or amount < 0:
                    errors.append(
                        f"Line {line_number}: Invalid estimated amount"
                    )

            # Check injury and police report fields
            for field in ["injury_reported", "police_report_filed"]:
                if label[field] is not None:
                    if not isinstance(label[field], bool):
                        errors.append(
                            f"Line {line_number}: Invalid value for {field}"
                        )
    # Display results
    print("\nCLAIMFAST DATASET VALIDATION")
    print("----------------------------")
    print(f"Total records: {len(records)}")

    if errors:
        print(f"\nFound {len(errors)} error(s):")

        for error in errors:
            print("-", error)

    else:
        print("All basic validation checks passed!")


if __name__ == "__main__":
    validate_dataset()