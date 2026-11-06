# CLAIMFAST
## AI-Powered Motor Insurance Claim Triage and Information Extraction

**Project:** ClaimFast  
**Domain:** Motor Insurance  
**Model:** `openai/gpt-oss-20b`  
**API Provider:** Groq  
**Temperature:** 0

---

# 1. Introduction

Motor insurance companies receive claim notifications through multiple communication channels, including emails, messages and customer support requests. These messages often contain unstructured information, making manual claim classification and information extraction time-consuming.

ClaimFast is an AI-powered claim triage and information extraction system designed to process motor insurance claim messages. It uses Large Language Models (LLMs) and prompt engineering techniques to identify claim categories, determine urgency and extract relevant claim information into structured JSON output.

The project investigates how different prompting strategies affect classification accuracy, extraction quality, response reliability and processing time.

# 2. Problem Statement

Motor insurance claim messages frequently contain incomplete information, informal language, spelling errors and ambiguous descriptions.

Manual processing can lead to delays, inconsistent classification and missed safety-critical information.

The objective of ClaimFast is to automate the initial analysis of claim messages and provide structured information that can support insurance claim handlers.

# 3. Project Objectives

- Classify incoming motor insurance messages into predefined claim categories.
- Identify the urgency of reported incidents.
- Extract relevant policy, vehicle, incident and location information.
- Generate structured JSON responses.
- Compare different prompt engineering strategies.
- Evaluate model accuracy, reliability, extraction performance and latency.
- Identify cases requiring human review.

# 4. System Output Fields

ClaimFast extracts the following nine fields:

| Field | Description |
|---|---|
| claim_type | Category of the insurance claim |
| urgency | High, medium or low |
| policy_number | Insurance policy identifier |
| incident_date | Date of the incident |
| location | Incident location |
| vehicle_reg_no | Vehicle registration number |
| estimated_amount_inr | Estimated financial damage in INR |
| injury_reported | Whether an injury was reported |
| police_report_filed | Whether a police report was filed |

The system uses predefined categories and structured output rules to maintain consistency.

# 5. Dataset Preparation

The project uses a synthetic and labelled dataset containing 80 motor insurance-related messages.

The dataset is divided into:

- Development dataset: 20 records.
- Held-out test dataset: 60 records.

The development dataset is used for prompt engineering and few-shot examples. The test dataset is reserved for evaluating prompt performance on unseen messages.

The dataset includes different claim categories, incomplete information, informal messages, spelling variations and ambiguous cases.

A labelling guide is maintained to establish consistent annotation rules. A review queue identifies ambiguous records requiring further inspection.

# 6. Prompt Engineering Strategies

Five prompting strategies were developed:

### 6.1 Zero-Shot Prompting

The model receives task instructions and a claim message without example inputs.

### 6.2 Few-Shot Prompting (K=3)

The model receives three labelled examples before processing the claim message.

### 6.3 Few-Shot Prompting (K=5)

Five labelled examples are provided to demonstrate the expected classification and extraction format.

### 6.4 Few-Shot Prompting (K=8)

Eight labelled examples are provided to expose the model to a wider range of claim scenarios.

### 6.5 Chain-of-Thought Prompting (CoT)

The model is instructed to provide a brief decision summary before producing its final structured JSON answer.

All five strategies use the same underlying model and temperature setting to support a controlled comparison.

# 7. Experimental Setup

The experiments use the Groq API with the `openai/gpt-oss-20b` model.

**Configuration:**

- Model: `openai/gpt-oss-20b`
- API provider: Groq
- Temperature: 0
- Development records: 20
- Held-out test records: 60
- Prompt strategies: 5
- Initial test runs per message: 1

The experiment runner records raw model responses, parsed JSON outputs, JSON validity, token usage and response latency.

A resume mechanism allows previously successful predictions to be skipped when an interrupted experiment is restarted.

The development experiment has been completed. The held-out test experiment has produced complete zero-shot results and partial K=3 results. The remaining test strategies are pending because the available daily API token allowance was reached.

# 8. Development Experiment Results

The following results were obtained from the completed 20-record development evaluation.

| Prompt strategy | Claim accuracy | Claim macro-F1 | Urgency accuracy | Average latency |
|---|---:|---:|---:|---:|
| Zero-shot | 95% | 0.9569 | 100% | 10.01 s |
| Few-shot K=3 | 95% | 0.9569 | 100% | 19.71 s |
| Few-shot K=5 | 95% | 0.9569 | 100% | 22.23 s |
| Few-shot K=8 | 100% | 1.0000 | 100% | 23.46 s |
| CoT | 100% | 1.0000 | 100% | 12.15 s |

All five strategies achieved 100% JSON validity and schema validity on the development dataset.

### 8.1 Development Findings

Few-shot K=8 and CoT achieved the highest claim classification performance on the development dataset.

CoT achieved the same classification accuracy as K=8 while having a lower average response latency.

Zero-shot prompting was the fastest among the five strategies, although its classification performance was slightly lower.

These findings are preliminary and require confirmation using the held-out test dataset.

# 9. Held-Out Test Evaluation

The held-out test dataset contains 60 messages. The first round of test experiments has produced the following results.

| Prompt strategy | Records expected | Records evaluated | Missing |
|---|---:|---:|---:|
| Zero-shot | 60 | 60 | 0 |
| Few-shot K=3 | 60 | 44 | 16 |
| Few-shot K=5 | 60 | 0 | 60 |
| Few-shot K=8 | 60 | 0 | 60 |
| CoT | 60 | 0 | 60 |

### 9.1 Zero-Shot Test Results

Zero-shot prompting completed all 60 held-out test messages.

| Metric | Result |
|---|---:|
| Claim type accuracy | 96.67% |
| Claim type macro-F1 | 0.9645 |
| Urgency accuracy | 98.33% |
| Urgency macro-F1 | 0.9765 |
| Extraction F1 | 0.8824 |
| JSON validity | 100% |
| Schema validity | 100% |
| Injury recall | 100% |
| High-urgency recall | 100% |
| Average latency | 15.44 seconds |

The results indicate strong classification performance and reliable structured JSON generation on the complete test dataset.

### 9.2 Few-Shot K=3 Test Results

Few-shot K=3 successfully processed 44 of the 60 test messages.

| Metric | Result |
|---|---:|
| Claim type accuracy | 95.45% |
| Claim type macro-F1 | 0.9381 |
| Urgency accuracy | 97.73% |
| Urgency macro-F1 | 0.9646 |
| Extraction F1 | 0.9134 |
| JSON validity | 100% |
| Schema validity | 100% |
| Injury recall | 100% |
| High-urgency recall | 100% |
| Average latency | 12.91 seconds |

These figures are based on 44 successfully evaluated records. They should not be directly compared with the complete 60-record zero-shot results as a definitive strategy ranking.

### 9.3 Initial Test Findings

Zero-shot achieved higher classification accuracy and macro-F1 than the partial K=3 evaluation.

K=3 achieved a higher extraction F1 on its available 44 records. However, the different evaluation sample sizes prevent a fair final comparison.

Both strategies achieved 100% JSON and schema validity on their successfully evaluated records.

The K=5, K=8 and CoT test experiments remain pending. Consequently, no final conclusion about the best overall prompting strategy can yet be established.

# 10. Error Analysis

Error analysis is maintained separately in `outputs/error_analysis.csv`.

The current analysis contains 51 field-level mismatches from the available zero-shot and K=3 predictions.

| Error category | Identified mismatches |
|---|---:|
| Classification errors | 4 |
| Urgency errors | 2 |
| Location normalization | 36 |
| Location omissions | 5 |
| Unsupported extraction | 2 |
| Injury extraction errors | 2 |
| Total | 51 |

These are field-level mismatches, not 51 distinct failed messages.

### 10.1 Classification Errors

Some messages were assigned an incorrect claim category.

For example, TEST018 was labelled `glass_only` but both zero-shot and K=3 predicted `not_a_claim`.

TEST051 was labelled `own_damage`, while zero-shot predicted `third_party_liability`.

These cases indicate the importance of clear category definitions and human review for ambiguous incidents.

### 10.2 Location Extraction Errors

Location extraction was a significant source of exact-match differences.

Examples include:

- Gold: `Dadar`; prediction: `near Dadar`.
- Gold: `Vashi`; prediction: `near Vashi`.
- Gold: `Workplace parking`; prediction: empty.
- Gold: `Parking space`; prediction: empty.

Some differences may be semantically equivalent, while others represent missing or unsupported information. Exact-match evaluation does not distinguish these cases.

### 10.3 Other Extraction Issues

Incident-date metrics require additional investigation. The zero-shot evaluation reported 100% exact match but zero precision, recall and F1. This may occur when there are no positive incident-date values available for evaluating extraction performance.

The gold labels and evaluation logic should be inspected before drawing conclusions about date extraction quality.

The current error analysis is preliminary and covers only the strategies with available successful test predictions. Additional strategies and manual verification remain necessary.

# 11. Production Recommendations

ClaimFast is intended to operate as an initial claim triage assistant rather than an autonomous claim approval system.

For a proposed deployment handling 5,000 messages per day, the production workflow should include:

- Automated classification and information extraction.
- Human review for injury-related and high-urgency incidents.
- Human review for ambiguous claim categories and multiple incidents.
- Verification of missing or contradictory policy and vehicle information.
- Monitoring of extraction quality, classification performance, latency and token consumption.
- Periodic evaluation against manually reviewed claim records.

Confidence thresholds should be established using validation data and calibrated before production deployment. The current experiment does not yet establish reliable numerical confidence thresholds.

The final prompt strategy should be selected after completing the full held-out comparison, including repeated-run consistency and cost analysis.

# 12. Limitations and Future Scope

Current limitations include:

- The dataset is synthetic and may not fully represent real-world claim communication.
- The held-out test comparison is incomplete.
- The initial test experiment uses one run per message rather than the three runs requested for consistency evaluation.
- Daily API token limits interrupted the experiment.
- Exact-match evaluation can penalise semantically equivalent location descriptions.
- Automated lexical hallucination checks require manual verification.
- Ambiguous messages may require human interpretation.
- Cost per 1,000 claims and run-to-run agreement have not yet been finalised.

Future improvements include:

- Completing all five test experiments.
- Performing repeated-run consistency testing.
- Improving extraction of ambiguous locations and incident dates.
- Introducing confidence-based human review.
- Expanding the dataset with more diverse claim scenarios.
- Evaluating additional LLMs.
- Integrating the system with insurance claim management workflows.

# 13. Conclusion

ClaimFast demonstrates how Large Language Models and prompt engineering can be applied to motor insurance claim triage and structured information extraction.

The completed development experiments show promising results across the five prompting strategies. Initial held-out testing demonstrates strong zero-shot classification performance, reliable JSON generation and promising extraction performance.

The partial K=3 results also provide useful evidence about the effect of few-shot examples on extraction quality.

However, the remaining prompt experiments, repeated-run evaluation and final cost analysis are necessary before selecting a production strategy.

ClaimFast is therefore a promising decision-support prototype, with human review retained for ambiguous and safety-critical cases.

---

**Experiment status:** Initial held-out test evaluation in progress.  
**Final strategy recommendation:** Pending completion of the remaining experiments.