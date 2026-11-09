# ClaimFast
### AI-Powered Motor Insurance Claim Triage and Information Extraction

ClaimFast is an LLM-powered motor insurance claim triage prototype that classifies incoming claim messages, determines urgency and extracts important information into structured JSON.

The project explores different prompt engineering strategies and evaluates their impact on classification accuracy, information extraction, output reliability, token consumption and response latency.

---

## 1. Project Overview

Motor insurance companies receive claim notifications through emails, messages and customer support channels. These messages are often unstructured, incomplete or ambiguous.

ClaimFast uses a Large Language Model (LLM) to automate the initial analysis of these messages and generate structured information that can assist insurance claim handlers.

**Key capabilities:**
- Motor insurance claim classification.
- Urgency detection.
- Structured information extraction.
- JSON output generation.
- Prompt engineering experimentation.
- Performance and reliability evaluation.

ClaimFast is designed as a decision-support system, not an autonomous claim approval or rejection system.

## 2. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| LLM | `openai/gpt-oss-20b` |
| API Provider | Groq |
| API Library | Groq Python SDK |
| Environment | Python Virtual Environment |
| IDE | Visual Studio Code |
| Data Format | JSONL |
| Evaluation | Python, Pandas, Scikit-learn |
| Configuration | python-dotenv |

## 3. System Architecture

The ClaimFast workflow consists of the following stages:

1. **Input:** Receive an unstructured motor insurance claim message.
2. **Prompt Construction:** Load the selected prompt strategy.
3. **LLM Processing:** Send the message to the Groq-hosted model.
4. **Classification:** Predict the claim category and urgency.
5. **Information Extraction:** Extract relevant policy, vehicle and incident details.
6. **JSON Generation:** Return the extracted information in a structured format.
7. **Validation:** Parse and validate the model response.
8. **Evaluation:** Compare predictions against labelled records.

## 4. Output Schema

ClaimFast extracts nine fields:

| Field | Description |
|---|---|
| `claim_type` | Category of the insurance claim |
| `urgency` | High, medium or low |
| `policy_number` | Insurance policy identifier |
| `incident_date` | Date of the incident |
| `location` | Incident location |
| `vehicle_reg_no` | Vehicle registration number |
| `estimated_amount_inr` | Estimated damage amount in INR |
| `injury_reported` | Whether an injury was reported |
| `police_report_filed` | Whether a police report was filed |

The system uses predefined categories and structured output rules to maintain consistency.

## 5. Dataset

The project uses a synthetic and labelled dataset containing 80 motor insurance-related messages.

| Dataset | Records | Purpose |
|---|---:|---|
| DEV | 20 | Prompt development and few-shot examples |
| TEST | 60 | Held-out performance evaluation |
| Total | 80 | Complete dataset |

The dataset includes different claim categories, incomplete information, informal messages, spelling variations and ambiguous scenarios.

The project also maintains:
- `labelling_guide.md` – annotation rules and category definitions.
- `review_queue.md` – records requiring additional inspection.

## 6. Prompt Engineering Strategies

Five principal prompting strategies were evaluated.

### A. Zero-Shot Prompting

The model receives task instructions, classification definitions, priority rules and an output schema without labelled examples.

### B. Few-Shot Prompting (K=3)

Three labelled examples are included in the prompt to demonstrate the expected classification and extraction behaviour.

### C. Few-Shot Prompting (K=5)

Five labelled examples are provided to expose the model to additional claim scenarios.

### D. Few-Shot Prompting (K=8)

Eight labelled examples are included to provide a wider range of examples.

### E. Chain-of-Thought (CoT) Prompting

A reasoning-guided prompt provides an internal decision checklist before generating the final JSON response.

The revised CoT version used in the final repeated-run evaluation generates JSON-only output. It does not use visible `<reasoning>` and `<answer>` tags.

An original CoT prompt was also evaluated separately.

## 7. Project Structure

```text
ClaimFast/
│
├── data/
│   ├── dev.jsonl
│   ├── test.jsonl
│   ├── labelling_guide.md
│   └── review_queue.md
│
├── prompts/
│   ├── zero_shot.txt
│   ├── few_shot_k3.txt
│   ├── few_shot_k5.txt
│   ├── few_shot_k8.txt
│   ├── cot.txt
│   └── cot_original_v1.txt
│
├── outputs/
│   ├── zero_shot_results.jsonl
│   ├── few_shot_k3_results.jsonl
│   ├── few_shot_k5_results.jsonl
│   ├── few_shot_k8_results.jsonl
│   ├── cot_results.jsonl
│   ├── test_zero_shot_v2_results.jsonl
│   ├── test_zero_shot_v2_run2_results.jsonl
│   ├── test_zero_shot_v2_run3_results.jsonl
│   ├── test_few_shot_k3_results.jsonl
│   ├── test_few_shot_k3_run2_results.jsonl
│   ├── test_few_shot_k3_run3_results.jsonl
│   ├── test_few_shot_k5_results.jsonl
│   ├── test_few_shot_k5_run2_results.jsonl
│   ├── test_few_shot_k5_run3_results.jsonl
│   ├── test_few_shot_k8_results.jsonl
│   ├── test_few_shot_k8_run2_results.jsonl
│   ├── test_few_shot_k8_run3_results.jsonl
│   ├── test_cot_controlled_results.jsonl
│   ├── test_cot_controlled_run2_results.jsonl
│   ├── test_cot_controlled_run3_results.jsonl
│   ├── test_repeated_run_summary.csv
│   ├── test_repeated_run_metrics.csv
│   ├── test_run_to_run_agreement.csv
│   ├── test_repeated_error_analysis.csv
│   ├── test_repeated_error_summary.csv
│   └── test_error_analysis_coverage.csv
│
├── report/
│   └── claimfast_report.md
│
├── src/
│   ├── validate_dataset.py
│   ├── validate_all_datasets.py
│   ├── run_prompts.py
│   ├── evaluate_results.py
│   ├── evaluate_test_results.py
│   ├── evaluate_all_runs.py
│   └── analyze_repeated_errors.py
│
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
├── test_groq.py
└── README.md
```

**Note:** The output directory contains additional evaluation files and confusion matrices beyond the representative files listed above.

## 8. Installation and Setup

### Prerequisites

- Python 3.12 or a compatible Python version.
- Visual Studio Code.
- Groq API account and API key.
- Git (optional, for version control).

### Step 1: Clone the repository

```bash
git clone <your-repository-url>
cd ClaimFast
```

If the project is already downloaded, open its folder in Visual Studio Code.

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own API key.

**Important:** Never upload your `.env` file or expose your API key publicly.

The `.env.example` file should contain only the variable name and a placeholder.

## 9. Running the Project

### Step 1: Validate the dataset

Run:

```bash
python src/validate_dataset.py
```

To validate both datasets:

```bash
python src/validate_all_datasets.py
```

### Step 2: Select a prompt strategy

Open:

```text
src/run_prompts.py
```

Configure the prompt and output settings.

The runner uses the following configuration variables:

```python
PROMPT_FILE = "zero_shot.txt"
PROMPT_NAME = "zero_shot_v2"
PROMPT_VERSION = "updated_classification_priority"
OUTPUT_STEM = "test_zero_shot_v2"
RUN_NUMBER = 1
```

Change these settings according to the prompt strategy and experiment you want to execute.

### Step 3: Execute the experiment

Run:

```bash
python src/run_prompts.py
```

The experiment runner:
- Loads the selected prompt.
- Reads the TEST dataset.
- Sends messages to the Groq API.
- Stores raw model responses.
- Parses JSON responses.
- Records token usage and latency.
- Saves results in JSONL format.
- Skips previously completed successful records when resuming an interrupted run.

**API availability:** Execution requires a valid API key and available model quota. Previously generated results can be evaluated without making additional API calls.

## 10. Evaluation

ClaimFast evaluates the generated predictions against the labelled dataset.

### Evaluation metrics

**Classification**
- Accuracy.
- Macro-F1.
- Confusion matrices.

**Information Extraction**
- Exact match.
- Precision.
- Recall.
- F1-score.

**Reliability**
- JSON validity.
- Run-to-run classification agreement.
- Full-output agreement.

**Performance**
- Average input tokens.
- Average output tokens.
- Average total tokens.
- Average response latency.

**Safety**
- Injury recall.
- High-urgency recall.

**Error Analysis**
- Classification errors.
- Urgency errors.
- Location omissions.
- Location wording mismatches.
- Formatting differences.
- Potential unsupported extraction.

### Evaluation commands

For individual evaluation:

```bash
python src/evaluate_results.py
```

For TEST evaluation:

```bash
python src/evaluate_test_results.py
```

For repeated-run evaluation:

```bash
python src/evaluate_all_runs.py
```

For repeated error analysis:

```bash
python src/analyze_repeated_errors.py
```

The evaluation scripts save their results in the `outputs/` directory.

## 11. Experimental Results

The final repeated-run evaluation covered three runs of 60 TEST records for each of the five principal strategies.

| Strategy | Claim Accuracy | Claim Macro-F1 | Extraction F1 |
|---|---:|---:|---:|
| Zero-Shot V2 | 98.33% | 98.48% | 87.80% |
| Few-Shot K=3 | 95.56% | 95.22% | 85.99% |
| Few-Shot K=5 | 95.00% | 95.01% | 86.05% |
| Few-Shot K=8 | 95.00% | 94.96% | 84.66% |
| Revised CoT | 96.11% | 95.82% | 91.73% |

### Key Findings

- Zero-Shot V2 achieved the highest claim classification accuracy.
- Revised CoT achieved the highest extraction F1.
- Few-shot prompting did not consistently improve classification performance.
- K=3 achieved the lowest average latency among the five repeated-run strategies.
- K=8 used the most tokens and had the highest average latency.
- Zero-Shot V2 achieved 100% classification agreement across its three runs.

These findings apply to the project's synthetic dataset and tested prompt versions.

## 12. Production Considerations

For a proposed workload of 5,000 claim messages per day, ClaimFast should operate as an initial triage assistant.

Recommended safeguards include:

- Human review for injury-related and high-urgency incidents.
- Human review for ambiguous or multiple-incident messages.
- Verification of missing policy and vehicle details.
- JSON validation before downstream processing.
- Monitoring of classification and extraction performance.
- Regular evaluation using manually reviewed claim records.
- Confidence threshold calibration using separate validation data.

The current experiment does not establish calibrated confidence thresholds or a verified production cost per 1,000 claims.

## 13. Limitations

- The dataset is synthetic and relatively small.
- Only 60 messages are available in the held-out TEST dataset.
- Results may not generalise to real-world insurance claims.
- Prompt versions have undergone revisions and must be interpreted accordingly.
- The original CoT strategy has only one completed TEST run.
- Exact-match extraction can penalise semantically equivalent wording.
- Potential unsupported extraction is a heuristic and requires manual verification.
- API quota limitations can interrupt new experiment execution.
- Production cost and throughput require further validation.

## 14. Future Scope

Potential improvements include:

- Expanding the dataset with more diverse claim scenarios.
- Improving location normalisation.
- Improving extraction of incident dates and missing information.
- Introducing confidence-based human review.
- Testing additional LLMs.
- Evaluating real-world insurance claim messages.
- Integrating with insurance claim management systems.
- Developing a user-friendly web interface.

## 15. Conclusion

ClaimFast demonstrates how Large Language Models and prompt engineering can be applied to motor insurance claim triage and structured information extraction.

The experiments show that zero-shot prompting can achieve strong classification performance, while reasoning-guided CoT can improve information extraction quality. Few-shot prompting does not necessarily guarantee better results as the number of examples increases.

The project provides a foundation for an AI-assisted claim triage workflow, with human review retained for ambiguous and safety-critical cases.

---

**Project:** ClaimFast  
**Domain:** Motor Insurance  
**Model:** `openai/gpt-oss-20b`  
**API Provider:** Groq  
**Language:** Python