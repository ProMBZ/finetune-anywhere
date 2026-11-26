# Dataset card: n8n workflow diagnosis and repair v0.2

## Summary
300 original synthetic chat examples for adapting Qwen3-1.7B to diagnose n8n workflow failures and produce conservative JSON Patch proposals.

| Split | Count |
|---|---:|
| Train | 200 |
| Validation | 50 |
| Evaluation | 50 |
| Total | 300 |

Training contains 25 cases for each category: expression, data mapping, item linking, node configuration, workflow structure, Code node, execution/API, and credentials.

## Coverage
- Expression syntax, node references, nested paths, date and string methods.
- Data mapping: missing fields, arrays, numeric and boolean conversion, nullish defaults.
- Item linking: .item, .first(), .all(), and Code node pairedItem.
- Node settings: HTTP method, request body, response format, timeout, Wait duration.
- Workflow graph: missing/dangling connections, IF true/false branch routing, Merge inputs.
- Code node: return item shape, execution mode, supported input API, item pairing.
- API execution: 429 retry settings, request payload keys/types, remote 5xx and DNS failures.
- Credential/security: expired or missing credentials, invalid keys, permissions, safe abstention.

## Construction and provenance
Examples are original synthetic fixtures. No customer workflow exports, production data, real credentials, or copied user records are included. Each prompt includes a small workflow excerpt, observed error, input sample where useful, and intended behavior.

## Evaluation protocol
The original 10 held-out examples are retained and 40 new cases are added. Evaluation remains separate from training and validation. Run the same prompt set and deterministic generation settings against the base and fine-tuned model.

## Known limitations
- Synthetic examples do not represent the frequency or full diversity of real n8n failures.
- Some cases use simplified workflow JSON fixtures and omit unrelated n8n metadata.
- A gold-patch match is strict and can penalize a different valid repair.
- Patch applicability is a structural JSON fixture check, not a live n8n execution test.
- A 1.7B adapter is not a substitute for current node documentation, execution data, or human review.
- Current documentation sources are listed in ../knowledge/sources.yaml; a complete runtime retrieval index is not included in this dataset.

## Safety
Credential and secret cases abstain. The model must never invent, reveal, or modify secret values. Do not automatically apply model-generated patches to a production workflow.
