# Dataset card: n8n workflow diagnosis and repair v0.3

## Summary

A controlled-vocabulary dataset for adapting Qwen3-1.7B to diagnose n8n workflow failures and propose conservative repairs.

| Split | Examples | Per category |
|---|---:|---:|
| Training | 640 | 80 |
| Validation | 128 | 16 |
| Held-out evaluation | 192 | 24 |
| **Total** | **960** | |

There are 48 canonical diagnosis codes: six in each of eight categories. Exact values are documented in [diagnosis-codes.json](diagnosis-codes.json). The same exact labels are used in all splits.

## Split policy

- Training retains 197 reviewed v0.2 training examples and 50 former validation examples after diagnosis-label normalization, plus 393 new or replacement synthetic scenarios.
- Validation contains 128 new cases.
- Evaluation contains 192 new cases, 24 per category and four per diagnosis code.
- The previous 50-case evaluation set is archived unchanged in archive/eval-v0.2.jsonl. It is not used for v0.3 training or evaluation. The new v0.3 evaluation prompts use unique case IDs, inputs and execution descriptions. Some underlying repair patterns necessarily recur because the diagnosis taxonomy is finite.

## Coverage and provenance

Expressions, data mapping, item linking, node configuration, workflow structure, Code node behavior, execution/API errors, and credentials/security. Examples are synthetic and contain no customer workflow exports, production records or real secrets. New examples vary node names, payloads, observed errors and workflow graphs around a documented root-cause taxonomy.

## Retrieval and training

The Colab notebook retrieves official n8n documentation for train, validation and evaluation prompts using the same local BM25 index and augmentation settings. Retrieved excerpts are runtime context, not memorized fine-tuning records. The notebook validates actual tokenized lengths before training.

## Limitations

- Synthetic test cases cannot represent the full distribution of production n8n failures.
- Patch applicability checks JSON Patch application and workflow graph structure; it does not execute a live n8n workflow.
- Gold-patch match is strict and may penalize alternative valid patches.
- 100% scores on this benchmark are not a guarantee of production reliability.
- External outages and credential problems require human review. Missing item lineage can use `.first()` only when a single upstream item is explicitly guaranteed; otherwise it requires review.

## Safety

Credential examples abstain. Do not commit workflow exports containing credentials, tokens, customer data or private URLs. Never automatically apply model-generated patches to production workflows.
