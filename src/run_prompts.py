
import os
import json
import time
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
from groq import Groq
from groq import RateLimitError, APIConnectionError, InternalServerError


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing. Check your .env file.")

client = Groq(api_key=api_key)

MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0
REQUEST_DELAY = 3
MAX_RETRIES = 3


# ==========================================
# 2. PROMPT CONFIGURATION
# ==========================================

PROMPT_FILES = [
    "zero_shot.txt",
    "few_shot_k3.txt",
    "few_shot_k5.txt",
    "few_shot_k8.txt",
    "cot.txt"
]

RUNS_PER_MESSAGE = 1


# ==========================================
# 3. LOAD PROMPTS
# ==========================================

def load_prompt(prompt_name, message):

    prompt_path = PROMPTS_DIR / prompt_name

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {prompt_path}"
        )

    template = prompt_path.read_text(encoding="utf-8")

    if "{message}" not in template:
        raise ValueError(
            f"{{message}} placeholder missing in {prompt_name}"
        )

    return template.replace("{message}", message)


# ==========================================
# 4. LOAD DATASET
# ==========================================

def load_dataset(dataset_name):

    dataset_path = DATA_DIR / dataset_name

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )

    records = []

    with dataset_path.open("r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}: {error}"
                    )

    if not records:
        raise ValueError("Dataset is empty.")

    return records


# ==========================================
# 5. GROQ API CALL WITH RETRIES
# ==========================================

def call_groq(prompt):

    for attempt in range(MAX_RETRIES):

        try:

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

            return {
                "raw_output": response.choices[0].message.content,
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "latency_seconds": round(latency, 4)
            }

        except (RateLimitError, APIConnectionError, InternalServerError) as error:

            if attempt == MAX_RETRIES - 1:
                raise

            wait_time = 5 * (attempt + 1)

            print(
                f"Temporary API error. "
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


# ==========================================
# 6. PARSE MODEL OUTPUT
# ==========================================

def parse_model_output(raw_output, prompt_name):

    try:

        if prompt_name == "cot.txt":

            if "<answer>" not in raw_output or "</answer>" not in raw_output:
                return None, False

            json_text = raw_output.split(
                "<answer>", 1
            )[1].split(
                "</answer>", 1
            )[0].strip()

        else:

            json_text = raw_output.strip()

            if json_text.startswith("```"):

                lines = json_text.splitlines()

                if lines and lines[0].startswith("```"):
                    lines = lines[1:]

                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]

                json_text = "\n".join(lines).strip()

        parsed_output = json.loads(json_text)

        if not isinstance(parsed_output, dict):
            return None, False

        return parsed_output, True

    except (json.JSONDecodeError, AttributeError, TypeError):

        return None, False


# ==========================================
# 7. OUTPUT FILE MANAGEMENT
# ==========================================

def get_output_path(prompt_name, dataset_name):

    dataset_prefix = Path(dataset_name).stem

    strategy_name = Path(prompt_name).stem

    return OUTPUTS_DIR / (
        f"{dataset_prefix}_{strategy_name}_results.jsonl"
    )


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

                if result.get("status") == "success":
                    completed.add(
                        (
                            result.get("record_id"),
                            result.get("run_number")
                        )
                    )

            except json.JSONDecodeError:
                continue

    return completed


def save_result(result, output_path):

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as file:

        file.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )


# ==========================================
# 8. MAIN EXPERIMENT RUNNER
# ==========================================

def run_experiment(
    dataset_name="dev.jsonl",
    start_index=0,
    record_limit=None
):

    records = load_dataset(dataset_name)

    if record_limit is None:
        selected_records = records[start_index:]
    else:
        selected_records = records[
            start_index:start_index + record_limit
        ]

    if not selected_records:
        raise ValueError("No records selected.")

    print("\n==========================================")
    print("       CLAIMFAST PROMPT EXPERIMENT")
    print("==========================================")

    print(f"Dataset: {dataset_name}")
    print(f"Records loaded: {len(selected_records)}")
    print(f"First record: {selected_records[0]['id']}")
    print(f"Last record: {selected_records[-1]['id']}")
    print(f"Prompt strategies: {len(PROMPT_FILES)}")
    print(f"Runs per message: {RUNS_PER_MESSAGE}")

    total_calls = (
        len(selected_records)
        * len(PROMPT_FILES)
        * RUNS_PER_MESSAGE
    )

    completed_calls = 0
    skipped_calls = 0
    failed_calls = 0

    print(f"Total planned calls: {total_calls}")

    for prompt_name in PROMPT_FILES:

        output_path = get_output_path(
            prompt_name,
            dataset_name
        )

        completed = load_completed_runs(output_path)

        print("\n" + "=" * 50)
        print(f"Running strategy: {prompt_name}")
        print("=" * 50)

        for record in selected_records:

            record_id = record["id"]
            message = record["message"]

            prompt = load_prompt(
                prompt_name,
                message
            )

            for run_number in range(1, RUNS_PER_MESSAGE + 1):

                run_key = (record_id, run_number)

                if run_key in completed:

                    skipped_calls += 1

                    print(
                        f"{record_id} | Run {run_number} | Already completed - skipped"
                    )

                    continue

                try:

                    metrics = call_groq(prompt)

                    parsed_output, json_valid = parse_model_output(
                        metrics["raw_output"],
                        prompt_name
                    )

                    result = {
                        "record_id": record_id,
                        "prompt_name": prompt_name,
                        "run_number": run_number,
                        "dataset": dataset_name,
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

                    completed.add(run_key)
                    completed_calls += 1

                    print(
                        f"{record_id} | "
                        f"Run {run_number} | "
                        f"JSON valid: {json_valid} | "
                        f"Latency: {metrics['latency_seconds']}s"
                    )

                except Exception as error:

                    failed_calls += 1

                    error_result = {
                        "record_id": record_id,
                        "prompt_name": prompt_name,
                        "run_number": run_number,
                        "dataset": dataset_name,
                        "timestamp": datetime.now().isoformat(),
                        "model": MODEL,
                        "status": "failed",
                        "error": str(error)
                    }

                    save_result(error_result, output_path)

                    print(
                        f"ERROR | {record_id} | "
                        f"Run {run_number} | {error}"
                    )

                time.sleep(REQUEST_DELAY)

    # ======================================
    # 9. EXPERIMENT SUMMARY
    # ======================================

    print("\n==========================================")
    print("          EXPERIMENT SUMMARY")
    print("==========================================")

    print(f"Planned API calls: {total_calls}")
    print(f"New successful calls: {completed_calls}")
    print(f"Previously completed calls skipped: {skipped_calls}")
    print(f"Failed calls: {failed_calls}")

    print(f"Results saved in: {OUTPUTS_DIR}")

    print("\nExperiment completed!")


# ==========================================
# 10. PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":

    run_experiment(
        dataset_name="test.jsonl",
        start_index=0,
        record_limit=None
    )
