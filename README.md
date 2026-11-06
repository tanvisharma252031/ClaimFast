# ClaimFast — AI-Powered Motor Insurance Claim Processing

## 1. Project Overview

ClaimFast is an AI-powered motor insurance claim processing system designed to analyse incoming claim messages, identify claim categories, assess urgency, and extract important information into a structured JSON format.

The project explores how Large Language Models (LLMs) and prompt engineering techniques can automate the initial stages of insurance claim triage.

The system accepts natural-language messages, including informal language and ambiguous descriptions, and converts them into structured information that can support insurance claim assessment.

## 2. Objectives

- Classify incoming messages into appropriate claim categories.
- Identify the urgency of reported incidents.
- Extract important claim information.
- Generate structured and machine-readable JSON responses.
- Compare different prompt engineering strategies.
- Evaluate classification accuracy, extraction performance, reliability, latency and token usage.
- Identify cases that require human review.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Groq API | LLM inference |
| GPT-OSS-20B | Language model |
| python-dotenv | Environment variable management |
| Pandas | Evaluation and data analysis |
| JSON / JSONL | Dataset and prediction storage |
| CSV | Evaluation results |

## 4. Project Structure

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
│   └── cot.txt
│
├── src/
│   ├── validate_dataset.py
│   ├── validate_all_datasets.py
│   ├── run_prompts.py
│   ├── evaluate_results.py
│   └── evaluate_test_results.py
│
├── outputs/
│   ├── Raw prediction files
│   ├── Evaluation summaries
│   ├── Confusion matrices
│   └── error_analysis.csv
│
├── report/
│   └── claimfast_report.md
│
├── demo/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## 5. Dataset

The project uses 80 labelled synthetic/anonymised motor insurance messages.

| Dataset | Records | Purpose |
|---|---:|---|
| Development | 20 | Prompt development and few-shot examples |
| Test | 60 | Held-out model evaluation |
| Total | 80 | Complete dataset |

The messages represent different claim situations, including:

- Own vehicle damage
- Theft
- Glass-only damage
- Third-party liability
- Total loss
- Non-claim messages

The dataset also includes informal language, ambiguous descriptions and different incident scenarios.

## 6. Prompt Engineering Strategies

Five prompt configurations are implemented.

### A. Zero-shot prompting

The model receives task instructions and an input message without examples.

### B. Few-shot prompting

The model receives labelled examples before processing the input.

Three configurations are evaluated:

- K = 3 examples
- K = 5 examples
- K = 8 examples

Examples are selected from the development dataset.

### C. Chain-of-Thought (CoT) prompting

The model is instructed to provide a brief decision summary before producing its final structured JSON answer.

The experiment compares all five configurations using the same underlying model.

## 7. Output Structure

ClaimFast extracts the following fields:

| Field | Description |
|---|---|
| claim_type | Category of the insurance claim |
| urgency | Priority level |
| policy_number | Insurance policy identifier |
| incident_date | Date of the incident |
| location | Incident location |
| vehicle_reg_no | Vehicle registration number |
| estimated_amount_inr | Estimated financial loss |
| injury_reported | Whether an injury was reported |
| police_report_filed | Whether a police report was filed |

The output is designed to follow a predefined JSON schema.

## 8. Installation and Setup

### Step 1: Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd ClaimFast
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

```bash
source .venv/bin/activate
```

**Windows**

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the API key

Create a `.env` file in the project root.

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own Groq API key.

Never upload your `.env` file or API key to GitHub.

## 9. Running the Project

### Validate datasets

```bash
python src/validate_all_datasets.py
```

### Run prompt experiments

```bash
python src/run_prompts.py
```

The experiment runner processes the configured dataset and prompt strategies. Successful predictions are stored in JSONL files.

The runner also supports resuming experiments by skipping previously successful records.

### Evaluate development results

```bash
python src/evaluate_results.py
```

### Evaluate held-out test results

```bash
python src/evaluate_test_results.py
```

Evaluation outputs are saved in the `outputs/` directory.

## 10. Evaluation Metrics

The project evaluates:

- Classification accuracy
- Macro precision, recall and F1-score
- Confusion matrices
- Field-level extraction exact match and F1-score
- JSON and schema validity
- Injury recall and high-urgency recall
- Average input and output tokens
- Average response latency
- Estimated hallucination indicators
- Run-to-run agreement (when repeated runs are available)

The hallucination indicator is based on lexical comparison and requires manual verification.

## 11. Current Experimental Status

The held-out test experiment has produced the following initial results:

| Strategy | Successful predictions |
|---|---:|
| Zero-shot | 60/60 |
| Few-shot K=3 | 44/60 |
| Few-shot K=5 | Pending |
| Few-shot K=8 | Pending |
| CoT | Pending |

Zero-shot currently achieves:

- Claim type accuracy: 96.67%
- Urgency accuracy: 98.33%
- Extraction F1: 0.8824
- JSON validity: 100%
- Schema validity: 100%

Few-shot K=3 has partial results. The final comparison will be updated after the remaining experiments are completed.

## 12. Limitations

- Model performance depends on the quality and clarity of the input message.
- Ambiguous and multi-incident messages may require human review.
- Exact-match evaluation can penalise semantically equivalent location descriptions.
- Automated hallucination detection requires manual verification.
- The current experiments use a limited synthetic/anonymised dataset.
- Production deployment requires additional testing, monitoring and human oversight.

## 13. Future Scope

- Complete repeated experimental runs.
- Improve extraction of ambiguous locations and incident dates.
- Introduce confidence-based human review.
- Expand the dataset with more diverse claim scenarios.
- Evaluate additional LLMs.
- Develop a production-ready claim triage interface.
- Integrate the system with insurance claim management workflows.

## 14. Conclusion

ClaimFast demonstrates how prompt engineering and Large Language Models can support automated motor insurance claim triage and structured information extraction.

By comparing zero-shot, few-shot and CoT prompting strategies, the project aims to identify an effective balance between accuracy, reliability, inference cost and response time.

The system is designed as a decision-support solution, with human review retained for ambiguous and high-risk cases.
