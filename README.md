# ClaimFast – AI-Powered Motor Insurance Claim Triage

## 1. Project Overview

ClaimFast is an LLM-powered motor insurance claim triage system designed to analyse incoming insurance claim messages, classify claim categories, determine urgency and extract important information into structured JSON.

Motor insurance companies receive claim notifications through emails, WhatsApp messages and customer support channels. These messages are often unstructured, incomplete, informal or ambiguous.

ClaimFast uses a Large Language Model (LLM) to automate the initial analysis of these messages and generate structured information that can assist insurance claim handlers.

The project focuses on comparing different prompt engineering strategies to identify their impact on classification accuracy, extraction quality, reliability, response time and token consumption.

## 2. Project Objectives

- Classify incoming messages into appropriate motor insurance claim categories.
- Identify the urgency of reported incidents.
- Extract important policy, vehicle and incident information.
- Generate structured JSON responses.
- Compare zero-shot, few-shot and chain-of-thought prompting strategies.
- Evaluate classification and extraction performance.
- Analyse errors, reliability, latency and token usage.
- Identify cases requiring human review.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Groq API | LLM inference |
| GPT-OSS-20B | Language model |
| Groq Python SDK | API integration |
| Pandas | Evaluation and data analysis |
| Scikit-learn | Performance metrics |
| JSON / JSONL | Dataset and prediction storage |
| CSV | Evaluation results |
| python-dotenv | Environment variable management |

## 4. System Architecture

The ClaimFast workflow consists of the following stages:

1. **Input:** Receive an unstructured motor insurance claim message.
2. **Prompt Construction:** Load the selected prompting strategy.
3. **LLM Processing:** Send the message to the Groq-hosted language model.
4. **Classification:** Identify the claim category and urgency.
5. **Information Extraction:** Extract policy, vehicle and incident details.
6. **JSON Generation:** Generate structured output.
7. **Validation:** Parse and validate the model response.
8. **Evaluation:** Compare predictions against labelled records.

## 5. Dataset

The project uses 80 labelled synthetic/anonymised motor insurance messages.

| Dataset | Records | Purpose |
|---|---:|---|
| Development | 20 | Prompt development and few-shot examples |
| Test | 60 | Held-out model evaluation |
| Total | 80 | Complete dataset |

The dataset includes different claim situations, such as:

- Own vehicle damage
- Third-party liability
- Theft
- Total loss
- Glass-only damage
- Non-claim messages

It also includes informal language, spelling variations, missing information and ambiguous incident descriptions.

The project includes:

- `dev.jsonl` – Development dataset.
- `test.jsonl` – Held-out test dataset.
- `labelling_guide.md` – Annotation rules and classification definitions.
- `review_queue.md` – Records requiring additional manual inspection.

## 6. Prompt Engineering Strategies

Five principal prompting configurations were evaluated.

### A. Zero-Shot Prompting

The model receives task instructions, classification definitions, priority rules and an output schema without labelled examples.

### B. Few-Shot Prompting (K=3)

Three labelled examples are included in the prompt to demonstrate the expected classification and extraction behaviour.

### C. Few-Shot Prompting (K=5)

Five labelled examples are provided to expose the model to additional claim scenarios.

### D. Few-Shot Prompting (K=8)

Eight labelled examples are included to provide a wider range of examples.

### E. Chain-of-Thought (CoT) Prompting

A reasoning-guided prompt uses a decision checklist before generating the final structured JSON response.

A revised CoT version was used in the final repeated-run evaluation. An original CoT version was also evaluated separately.

## 7. Output Structure

ClaimFast extracts nine fields from each insurance claim message.

| Field | Description |
|---|---|
| `claim_type` | Insurance claim category |
| `urgency` | High, medium or low |
| `policy_number` | Insurance policy identifier |
| `incident_date` | Date of the incident |
| `location` | Incident location |
| `vehicle_reg_no` | Vehicle registration number |
| `estimated_amount_inr` | Estimated damage amount in INR |
| `injury_reported` | Whether an injury was reported |
| `police_report_filed` | Whether a police report was filed |

If information is not available in the original message, the model is instructed to return null rather than guess.

## 8. Project Structure

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
├── src/
│   ├── validate_dataset.py
│   ├── validate_all_datasets.py
│   ├── run_prompts.py
│   ├── evaluate_results.py
│   ├── evaluate_test_results.py
│   ├── evaluate_all_runs.py
│   ├── analyze_repeated_errors.py
│   └── run_prompts_before_controlled_cot.py
│
├── outputs/
│   ├── Raw model predictions
│   ├── Evaluation summaries
│   ├── Confusion matrices
│   ├── Repeated-run metrics
│   └── Error analysis files
│
├── report/
│   └── claimfast_report.md
│
├── .env.example
├── .gitignore
├── requirements.txt
├── test_groq.py
└── README.md
```

## 9. Installation and Setup

### Prerequisites

- Python 3.12 or compatible Python version.
- Groq API account and API key.
- Visual Studio Code (recommended).
- Git (for version control).

### Step 1: Clone the repository

```bash
git clone https://github.com/tanvisharma252031/ClaimFast.git
cd ClaimFast
```

### Step 2: Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 3: Activate the environment

For macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows:

```bash
.venv\Scripts\activate
```

### Step 4: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Configure the API key

Create a `.env` file in the project root.

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own Groq API key.

**Important:** Never upload `.env` or expose your API key publicly. The `.env.example` file should contain only a placeholder.

## 10. Running the Project

### Step 1: Validate the datasets

Validate the development dataset:

```bash
python src/validate_dataset.py
```

Validate both datasets:

```bash
python src/validate_all_datasets.py
```

### Step 2: Run prompt experiments

The experiment runner uses environment variables to select the prompt configuration.

For example, to run Zero-Shot:

```bash
PROMPT_FILE=zero_shot.txt PROMPT_NAME=zero_shot_v2 OUTPUT_STEM=test_zero_shot_v2 RUN_NUMBER=1 python src/run_prompts.py
```

For Few-Shot K=3:

```bash
PROMPT_FILE=few_shot_k3.txt PROMPT_NAME=few_shot_k3 OUTPUT_STEM=test_few_shot_k3 RUN_NUMBER=1 python src/run_prompts.py
```

For Few-Shot K=5:

```bash
PROMPT_FILE=few_shot_k5.txt PROMPT_NAME=few_shot_k5 OUTPUT_STEM=test_few_shot_k5 RUN_NUMBER=1 python src/run_prompts.py
```

For Few-Shot K=8:

```bash
PROMPT_FILE=few_shot_k8.txt PROMPT_NAME=few_shot_k8 OUTPUT_STEM=test_few_shot_k8 RUN_NUMBER=1 python src/run_prompts.py
```

For revised CoT:

```bash
PROMPT_FILE=cot.txt PROMPT_NAME=cot_revised OUTPUT_STEM=test_cot_controlled RUN_NUMBER=1 python src/run_prompts.py
```

To perform the second and third runs, change `RUN_NUMBER` to `2` and `3`.

The experiment runner:

- Loads the selected prompt.
- Reads the TEST dataset.
- Sends messages to the Groq API.
- Stores raw model responses.
- Parses JSON responses.
- Records token usage and latency.
- Saves predictions in JSONL format.
- Supports resuming interrupted experiments by skipping previously completed successful records.

Previously generated results can be evaluated without making additional API calls.

## 11. Evaluation Metrics

The project evaluates the following dimensions.

### Classification Performance

- Accuracy
- Macro precision
- Macro recall
- Macro-F1
- Confusion matrices

### Information Extraction

- Field-level exact match
- Precision
- Recall
- F1-score

### Reliability

- JSON validity
- Run-to-run classification agreement
- Full-output agreement

### Performance

- Average input tokens
- Average output tokens
- Average total tokens
- Average response latency

### Safety

- Injury recall
- High-urgency recall

### Error Analysis

- Classification errors
- Urgency errors
- Location omissions
- Location wording mismatches
- Formatting differences
- Potential unsupported extraction

The hallucination indicator is heuristic-based and requires manual verification.

### Evaluation Commands

Evaluate development results:

```bash
python src/evaluate_results.py
```

Evaluate held-out test results:

```bash
python src/evaluate_test_results.py
```

Evaluate repeated runs:

```bash
python src/evaluate_all_runs.py
```

Perform repeated error analysis:

```bash
python src/analyze_repeated_errors.py
```

Evaluation files are saved in the `outputs/` directory.

## 12. Experimental Results

The final repeated-run evaluation covered three runs of 60 held-out TEST records for each of the five principal prompting strategies.

The following table presents the mean results across the three runs.

| Prompt Strategy | Claim Accuracy | Claim Macro-F1 | Extraction F1 |
|---|---:|---:|---:|
| Zero-Shot V2 | 98.33% | 98.48% | 87.80% |
| Few-Shot K=3 | 95.56% | 95.22% | 85.99% |
| Few-Shot K=5 | 95.00% | 95.01% | 86.05% |
| Few-Shot K=8 | 95.00% | 94.96% | 84.66% |
| Revised CoT | 96.11% | 95.82% | 91.73% |

### Key Findings

- Zero-Shot V2 achieved the highest claim classification accuracy.
- Revised CoT achieved the highest extraction F1-score.
- Few-shot prompting did not consistently improve classification performance.
- K=3 had the lowest mean response latency among the five strategies.
- K=8 consumed the highest average number of tokens and had the highest mean response latency.
- Zero-Shot V2 achieved 100% run-to-run classification agreement.

These results demonstrate that adding more examples does not necessarily improve model performance.

## 13. Production Considerations

For a proposed workload of 5,000