# n8n workflow diagnosis and repair

Fine-tune the starter repository's Qwen3 1.7B text model to diagnose common n8n workflow failures and return minimal, structured repair patches.

This recipe reuses the repository's Unsloth Colab pipeline, dataset validation, base-versus-tuned evaluation, evidence capture and GGUF Q4_K_M export. It does not replace the starter infrastructure.

## Task

The model receives a compact JSON case containing:
- an n8n workflow excerpt (nodes, parameters and connections),
- the observed execution error,
- optional representative input data,
- the intended behavior.

It returns exactly one JSON object with:
- `category` and `diagnosis_code`,
- `faulty_node` and `root_cause`,
- a minimal JSON Patch-style `patch`,
- `needs_human_review`.

Supported categories: expressions, data mapping, item linking, node configuration, workflow structure, Code node output, and execution/API failures. Authentication and external-service failures must be identified without inventing or modifying secrets.

## Important scope and safety

This is a focused competition prototype, not a claim to repair every possible n8n workflow. The dataset uses small, explicit workflow fixtures with known expected patches. The benchmark measures structured diagnosis and whether proposed patches can be applied to those fixtures. It does not claim that a structurally applicable patch has been executed successfully in a live n8n instance.

Never commit real workflow exports containing credentials, tokens, customer data or private URLs. Use synthetic examples and redact all secrets. Do not automatically apply patches to production workflows.

## Dataset

Original, synthetic examples created for this project. The splits are separate:
- `data/train.jsonl`: 40 training examples
- `data/valid.jsonl`: 10 validation examples
- `data/eval.jsonl`: 10 held-out evaluation examples

Keep the evaluation split untouched until final scoring. Review every example and every expected patch before training.

## Train in Colab

[Open the prepared n8n repair Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb). Select a T4 GPU when available and run the cells in order. The notebook clones this fork and the project branch, not the organizer's upstream main branch. The notebook uses:
- base model alias: `starter` (Qwen3-1.7B),
- Unsloth 4-bit loading and LoRA adapters,
- the recipe's pinned configuration,
- held-out base-versus-fine-tuned outputs,
- GGUF Q4_K_M export.

## Evaluation

The notebook imports `scripts/score_n8n_repairs.py` and reports JSON validity, schema validity, category/diagnosis accuracy, patch match, patch applicability and safe-abstention behavior.

A patch that applies to the sample JSON fixture is not proof of runtime correctness. A later live n8n test is required before claiming an end-to-end workflow repair.

## License and provenance

All examples in this directory are original synthetic cases authored for this project and released under the repository's MIT license. They are not copied from user workflows or customer records.
