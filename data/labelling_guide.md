# ClaimFast – Dataset Labelling Guide

## 1. Project Objective

ClaimFast is a motor-insurance First Notice of Loss (FNOL) classification and information extraction system.

Its purpose is to analyse incoming customer messages, identify the type and urgency of an insurance claim, and extract relevant information into a structured JSON output.

The dataset contains 80 labelled messages:

- **20 development messages:** Used for prompt development and few-shot examples.
- **60 test messages:** Held out for final evaluation and never used for prompt tuning.

All messages should be synthetic or anonymised. Do not include real names, phone numbers, or actual policy numbers.

---

## 2. General Labelling Principles

Follow these principles for every record:

1. Label only information explicitly supported by the message.
2. Never invent or assume missing information.
3. Use `null` when a field is unknown, unavailable, or not mentioned.
4. Preserve the meaning of Hinglish messages and messages containing spelling mistakes.
5. Apply the same labelling rules consistently across all records.
6. Do not use test records when developing or modifying prompts.
7. Do not confuse a customer's general insurance enquiry with an actual claim.
8. When a message contains multiple incidents, identify and document the ambiguity rather than silently combining unrelated details.

---

## 3. Claim Type Definitions

The `claim_type` field must contain exactly one of the following values.

| Claim Type | Definition |
|---|---|
| `own_damage` | Damage to the insured person's own vehicle. |
| `third_party_liability` | Damage or injury involving another person or their property, for which the insured may be liable. |
| `theft` | Theft of the insured vehicle or property associated with the incident. |
| `total_loss` | The vehicle is explicitly declared or described as a total loss. |
| `glass_only` | Damage is limited to vehicle glass, such as a windshield or window. |
| `not_a_claim` | The message is unrelated to reporting an insurance claim. |

### Important classification rules

- A damaged vehicle with no stated third-party involvement generally falls under `own_damage`.
- If another person or their property is involved in a liability incident, use `third_party_liability`.
- If a vehicle or property has been stolen, use `theft`, even if a window was broken during the theft.
- If only the windshield or vehicle glass is damaged, use `glass_only`.
- Do not classify a vehicle as `total_loss` merely because the damage appears expensive. It must be explicitly described as a total loss.
- Policy renewals, certificate requests, premium enquiries, and general insurance questions should be classified as `not_a_claim`.

---

## 4. Urgency Classification

The `urgency` field must contain exactly one of these values:

- `high`
- `medium`
- `low`

### High urgency

Assign `high` when:

- An injury is reported.
- The vehicle is undrivable and unsafe.

### Medium urgency

Assign `medium` when:

- An incident has occurred and requires claim attention.
- The message describes vehicle damage but does not meet the high-urgency conditions.
- The vehicle is undrivable, but the message does not establish that it is unsafe and no injury is reported.

### Low urgency

Assign `low` when:

- The matter is routine or non-urgent.
- The message is a general insurance enquiry or administrative request rather than an active incident.

### Important rule

Do not automatically assign high urgency simply because a vehicle cannot be driven.

For example:

"My car hit a divider. Nobody was injured, but the vehicle cannot be driven."

The correct urgency is `medium` because the message does not establish that the vehicle is unsafe.

---

## 5. Field-Level Extraction Rules

ClaimFast must extract exactly nine fields.

### 5.1 claim_type

- Select one of the six permitted claim categories.
- Do not create new categories.
- Use the definitions in Section 3.

### 5.2 urgency

- Select `high`, `medium`, or `low`.
- Apply the rules in Section 4.
- Do not infer injuries or unsafe conditions.

### 5.3 policy_number

- Extract the policy number exactly as stated.
- Store it as a string.
- If it is absent or unclear, use `null`.
- Never generate a policy number.

### 5.4 incident_date

- Extract the date of the incident.
- Convert an unambiguous date into ISO format: `YYYY-MM-DD`.
- If the date is missing, incomplete, or ambiguous, use `null`.
- Do not assume a year when it is not established by the message or its context.

Example:

`21 October 2026` → `2026-10-21`

### 5.5 location

- Extract the incident location as free text.
- Preserve the location information given in the message.
- Do not invent missing addresses, cities, or landmarks.
- If the location is not mentioned, use `null`.

### 5.6 vehicle_reg_no

- Extract the vehicle registration number.
- Convert it to uppercase.
- Remove spaces.
- If it is missing or unclear, use `null`.

Example:

`mh 02 ab 1234` → `MH02AB1234`

### 5.7 estimated_amount_inr

- Extract the estimated damage or claim amount in Indian rupees.
- Store the result as an integer.
- Convert shorthand amounts when the value is unambiguous.

Examples:

- `₹45,000` → `45000`
- `45k` → `45000`
- `₹1.5 lakh` → `150000`

If the amount is absent, unclear, or cannot be reliably converted, use `null`.

### 5.8 injury_reported

This field must contain:

- `true` – An injury is explicitly reported.
- `false` – The message explicitly states that nobody was injured.
- `null` – Injury information is missing or unknown.

Do not assume that nobody was injured simply because the message does not mention injuries.

### 5.9 police_report_filed

This field must contain:

- `true` – The message explicitly confirms that a police report or complaint was filed.
- `false` – The message explicitly states that no police report or complaint was filed.
- `null` – Police-report information is missing or unknown.

Do not assume a police report was filed merely because theft or an accident occurred.

---

## 6. Handling Special and Difficult Messages

The dataset must include different real-world message patterns.

### 6.1 Hinglish messages

Understand the meaning of the message rather than relying on perfect English.

Example:

"Kal raat meri car ka accident hua, front bumper damage ho gaya."

Interpretation:
- An accident occurred.
- The insured vehicle's bumper was damaged.
- Classify according to the remaining information in the message.

### 6.2 Spelling mistakes and informal language

Customers may use abbreviations, incorrect spelling, incomplete sentences, or informal language.

Example:

"my vehical got damged ystrday."

Do not reject the message simply because it contains spelling errors.

### 6.3 Multiple incidents in one message

A customer may describe two separate incidents in a single message.

Rules:

- Identify that multiple incidents are being described.
- Do not combine unrelated dates, locations, amounts, or vehicle details into one incident.
- Flag the record for human review when one structured output cannot represent the incidents without losing important information.
- Do not invent a priority rule that is not supported by the message.

### 6.4 Forwarded email chains

Messages may contain forwarded emails, replies, and older conversations.

Rules:

- Identify the actual incident being reported.
- Avoid treating old email content as a new incident.
- Do not mix information from different incidents.
- If the latest message and forwarded content conflict, flag the record for human review.

### 6.5 Missing information

Customers may report an accident without providing all required details.

Rules:

- Extract the available information.
- Use `null` for missing fields.
- Do not fill gaps using assumptions.

### 6.6 Ambiguous theft and glass damage

Example:

"My car window was broken and my stereo is missing."

This is a theft-related incident because property has been reported missing. Use `theft`.

If the message only says that the car window was broken and nothing was stolen, use `glass_only`.

### 6.7 Non-claim messages

Examples include:

- Requests for insurance renewal certificates.
- Questions about policy premiums.
- Requests for policy documents.
- General enquiries about insurance coverage.

These should be classified as `not_a_claim`.

---

## 7. Dataset Composition

The complete dataset must contain 80 labelled messages.

| Dataset | Number of Records | Purpose |
|---|---:|---|
| Development | 20 | Prompt development and few-shot examples |
| Test | 60 | Final performance evaluation |
| **Total** | **80** | **Complete dataset** |

The dataset should include:

- All six claim types.
- High, medium, and low urgency cases.
- Messages with missing information.
- Hinglish messages.
- Messages with spelling mistakes.
- Multiple-incident messages.
- Forwarded email chains.
- Ambiguous theft and glass-damage cases.
- Non-claim messages.

The development set should contain examples covering every claim type so that few-shot prompts can be constructed meaningfully.

The test set must remain separate and untouched during prompt development.

---

## 8. Human Review and Label Validation

To improve dataset quality:

1. Review every labelled message against this guide.
2. Ask a second person to independently review at least 20 records.
3. Compare the two sets of labels.
4. Record disagreements.
5. Resolve disagreements using the definitions in this guide.
6. Update the guide if a genuinely unclear rule needs clarification.
7. Keep a record of corrections made to the dataset.

Do not silently change labels simply to improve model performance.

---

## 9. Labelled Output Format

Each message must have a corresponding structured JSON label.

The output must contain exactly these nine fields:

```json
{
  "claim_type": "own_damage",
  "urgency": "medium",
  "policy_number": null,
  "incident_date": "2026-10-21",
  "location": "Mumbai",
  "vehicle_reg_no": "MH02AB1234",
  "estimated_amount_inr": 45000,
  "injury_reported": false,
  "police_report_filed": null
}
```

The example above is illustrative. Actual labels must be based on the content of each message.

### JSON requirements

- Use valid JSON.
- Use double quotes for keys and string values.
- Use `null`, not `"null"`.
- Use `true` and `false`, not strings.
- Do not add explanatory text outside the JSON object.
- Do not add extra fields.
- Do not omit any of the nine required fields.

---

## 10. Final Quality Checklist

Before accepting a labelled record, verify:

- [ ] The claim type belongs to the permitted categories.
- [ ] Urgency follows the defined rules.
- [ ] Policy number is extracted only when available.
- [ ] Incident date uses ISO format when unambiguous.
- [ ] Location is not invented.
- [ ] Vehicle registration is uppercase and contains no spaces.
- [ ] Estimated amount is an integer or `null`.
- [ ] Injury status is correctly represented as true, false, or null.
- [ ] Police-report status is correctly represented as true, false, or null.
- [ ] All nine fields are present.
- [ ] The JSON is valid.
- [ ] No real personal or confidential information is included.
- [ ] Test records have not been used for prompt tuning.

**End of Labelling Guide**