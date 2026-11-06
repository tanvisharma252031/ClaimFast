# ClaimFast – Semantic Label Review Queue

## Objective

Identify records where the assigned labels may be ambiguous, incomplete, or inconsistent with the labelling guide.

These records require human review before the dataset is treated as final ground truth.

## Records Requiring Review

| Record ID | Potential issue | Review required |
|---|---|---|
| TEST012 | Third-party damage and own-vehicle damage | Confirm primary claim classification |
| TEST041 | Other driver's responsibility is uncertain | Confirm classification rule |
| TEST043 | Two separate incidents in one message | Flag for human review; one output cannot represent both incidents |
| TEST048 | Possible theft is uncertain | Confirm whether theft can be established |
| DEV013 | Hit-and-run involving damage to insured vehicle | Confirm own-damage classification |

## Review Rules

1. Do not automatically change labels.
2. Refer to the labelling guide for every decision.
3. Record the reason for each correction.
4. Flag messages that cannot be reliably represented by the nine-field schema.
5. Keep the original message unchanged.
6. Do not use the held-out test set to tune prompts.

## Review Status

Pending manual review.
