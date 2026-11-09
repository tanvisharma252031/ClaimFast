
import os
import json
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from groq import (
    Groq,
    RateLimitError,
    APIConnectionError,
    InternalServerError
)


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. Check your .env file."
    )

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0
REQUEST_DELAY = 3

DATASET_NAME = "test.jsonl"

# Configure the prompt through terminal environment variables.
PROMPT_FILE = os.getenv("PROMPT_FILE", "zero_shot.txt")
PROMPT_NAME = os.getenv("PROMPT_NAME", "zero_shot_v2")
PROMPT_VERSION = os.getenv(
    "PROMPT_VERSION",
    "updated_classification_priority"
)
OUTPUT_STEM = os.getenv(
    "OUTPUT_STEM",
    "test_zero_shot_v2"
)

# Select run using terminal: RUN_NUMBER=2 or RUN_NUMBER=3
RUN_NUMBER = int(os.getenv("RUN_NUMBER", "1"))

if RUN_NUMBER not in (1, 2, 3):
    raise ValueError("RUN_NUMBER must be 1, 2, or 3.")

# Each run gets a separate output file.
if RUN_NUMBER == 1:
    OUTPUT_FILENAME = f"{OUTPUT_STEM}_results.jsonl"
else:
    OUTPUT_FILENAME = (
        f"{OUTPUT_STEM}_run{RUN_NUMBER}_results.jsonl"
    )

client = Groq(api_key=API_KEY)


# ==========================================
# 2. LOAD PROMPT
# ==========================================

def load_prompt(message):

    prompt_path = PROMPTS_DIR / PROMPT_FILE

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt not found: {prompt_path}"
        )

    template = prompt_path.read_text(encoding="utf-8")

    if "{message}" not in template:
        raise ValueError(
            f"{{message}} placeholder missing in {PROMPT_FILE}"
        )

    return template.replace("{message}", message)


# ==========================================
# 3. LOAD TEST DATASET
# ==========================================

def load_dataset():

    dataset_path = DATA_DIR / DATASET_NAME

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    records = []

    with dataset_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                )

    if len(records) != 60:
        raise ValueError(
            f"Expected 60 TEST records, found {len(records)}."
        )

    return records


# ==========================================
# 4. CALL GROQ API
# ==========================================

def call_groq(prompt):

    start_time = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=TEMPERATURE
    )

    latency = time.perf_counter() - start_time

    usage = response.usage

    return {
        "raw_output": response.choices[0].message.content or "",
        "input_tokens": usage.prompt_tokens if usage else None,
        "output_tokens": usage.completion_tokens if usage else None,
        "latency_seconds": round(latency, 4)
    }


# ==========================================
# 5. PARSE JSON OUTPUT
# ==========================================

def parse_model_output(raw_output):

    if not isinstance(raw_output, str):
        return None, False

    text = raw_output.strip()

    # Remove Markdown code fences if present.
    if text.startswith("```"):

        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    try:

        parsed = json.loads(text)

        if isinstance(parsed, dict):
            return parsed, True

    except json.JSONDecodeError:
        pass

    # Try extracting a JSON object from surrounding text.
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:

        try:

            parsed = json.loads(text[start:end + 1])

            if isinstance(parsed, dict):
                return parsed, True

        except json.JSONDecodeError:
            pass

    return None, False


# ==========================================
# 6. LOAD COMPLETED RECORDS
# ==========================================

def load_completed_runs(output_path):

    completed = set()

    if not output_path.exists():
        return completed

    with output_path.open("r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            try:

                result = json.loads(line)

                if (
                    result.get("status") == "success"
                    and result.get("run_number") == RUN_NUMBER
                ):

                    completed.add(result.get("record_id"))

            except json.JSONDecodeError:
                continue

    return completed


# ==========================================
# 7. SAVE RESULTS
# ==========================================

def save_result(result, output_path):

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


# ==========================================
# 8. RUN EXPERIMENT
# ==========================================

def run_experiment():

    records = load_dataset()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / OUTPUT_FILENAME

    completed = load_completed_runs(output_path)

    print("\n==========================================")
    print(f"       CLAIMFAST {PROMPT_NAME.upper()}")
    print("==========================================")

    print(f"Dataset: {DATASET_NAME}")
    print(f"Records: {len(records)}")
    print(f"Prompt: {PROMPT_FILE}")
    print(f"Model: {MODEL}")
    print(f"Temperature: {TEMPERATURE}")
    print(f"Run number: {RUN_NUMBER}")
    print(f"Already completed: {len(completed)}/60")
    print(f"Output: {OUTPUT_FILENAME}")

    new_successes = 0
    skipped = 0
    failures = 0

    for record in records:

        record_id = record["id"]
        message = record["message"]

        if record_id in completed:

            skipped += 1

            print(
                f"{record_id} | Already completed — skipped"
            )

            continue

        try:

            prompt = load_prompt(message)

            metrics = call_groq(prompt)

            parsed_output, json_valid = parse_model_output(
                metrics["raw_output"]
            )

            result = {
                "record_id": record_id,
                "prompt_name": PROMPT_NAME,
                "prompt_version": PROMPT_VERSION,
                "run_number": RUN_NUMBER,
                "dataset": DATASET_NAME,
                "timestamp": datetime.now().isoformat(),
                "model": MODEL,
                "temperature": TEMPERATURE,
                "raw_output": metrics["raw_output"],
                "parsed_output": parsed_output,
                "json_valid": json_valid,
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "latency_seconds": metrics["latency_seconds"],
                "status": "success"
            }

            save_result(result, output_path)

            completed.add(record_id)
            new_successes += 1

            print(
                f"{record_id} | "
                f"JSON valid: {json_valid} | "
                f"Latency: {metrics['latency_seconds']}s"
            )

            time.sleep(REQUEST_DELAY)

        except RateLimitError as error:

            failures += 1

            print("\nGROQ RATE LIMIT REACHED.")
            print("Stopping safely. Saved results are preserved.")
            print(error)

            break

        except (APIConnectionError, InternalServerError) as error:

            failures += 1

            print(f"\nAPI ERROR at {record_id}: {error}")
            print("Stopping safely. Re-run later to resume.")

            break

        except Exception as error:

            failures += 1

            print(
                f"ERROR | {record_id} | "
                f"{type(error).__name__}: {error}"
            )

    print("\n==========================================")
    print("          RUN SUMMARY")
    print("==========================================")

    print(f"Total TEST records: {len(records)}")
    print(f"New successful calls: {new_successes}")
    print(f"Skipped records: {skipped}")
    print(f"Errors: {failures}")
    print(f"Completed: {len(completed)}/60")
    print(f"Saved to: {output_path}")


# ==========================================
# 9. MAIN
# ==========================================

if __name__ == "__main__":

    run_experiment()
