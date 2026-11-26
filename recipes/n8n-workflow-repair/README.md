# n8n workflow diagnosis and repair

Fine-tune the starter repository's Qwen3 1.7B text model to diagnose n8n workflow failures and return minimal, structured repair proposals.

This recipe reuses the starter repository's Unsloth Colab pipeline, validation, base-versus-fine-tuned evaluation, evidence capture and GGUF Q4_K_M export. It does not replace the starter infrastructure.

## Task

The model receives a compact JSON case containing:
- an n8n workflow excerpt (nodes, parameters and connections),
- the observed execution error,
- representative input data when available,
- the intended behavior,
- version or documentation context when available.

It returns exactly one JSON object with:
- category and diagnosis_code,
- faulty_node and root_cause,
- a minimal JSON Patch-style patch,
- needs_human_review.

The eight categories are expressions, data mapping, item linking, node configuration, workflow structure, Code node output, execution/API failures, and credentials/security.

## Dataset v0.2

The recipe now contains **300 synthetic cases**:

| Split | Cases | Purpose |
|---|---:|---|
| Training | 200 | Adapter training |
| Validation | 50 | Validation loss and checkpoint selection |
| Held-out evaluation | 50 | Base-versus-fine-tuned comparison |
| **Total** | **300** | |

The training split is balanced at 25 cases per category. Validation and evaluation also cover all eight categories. The original 10 held-out cases remain in the evaluation split; the added 40 cases broaden coverage.

The examples include expression references and syntax, nested and typed data mapping, item linking and pairedItem, HTTP Request and Wait settings, missing or misrouted workflow connections, Merge inputs, Code node item shape/pairing, API rate limits and payload validation, external outages, and credential-related abstention.

Run the recipe validator before training:

```bash
python scripts/validate_n8n_dataset.py
```

Keep data/eval.jsonl untouched. Do not move evaluation cases into training or repeatedly tune against their expected answers.

## Important scope and safety

This is a focused competition prototype, not a claim to repair every n8n workflow. Examples are synthetic and deliberately compact. The benchmark checks structured diagnosis, agreement with one authored reference, and whether proposed patches apply to the fixture. A structurally applicable patch is **not** proof that it executes successfully in a live n8n instance.

Credential and secret cases must abstain. Ambiguous item linking and external infrastructure failures must not trigger guessed edits. Never commit real workflow exports containing credentials, tokens, customer data or private URLs. Do not automatically apply patches to production workflows.

The Colab notebook now fetches the official Markdown sources listed in knowledge/sources.yaml, records successes/failures, chunks the pages, and builds a lightweight local BM25 retrieval index. Retrieved excerpts are passed to both the base and fine-tuned models on the same evaluation prompts. This knowledge is retrieved at inference/evaluation time; it is **not** part of the fine-tuning dataset and is not memorised by the adapter. The docs manifest is included in the run evidence ZIP.

## Train and compare in Colab

[Open the prepared n8n repair Colab notebook](https://colab.research.google.com/github/ProMBZ/finetune-anywhere/blob/project-01-n8n-workflow-repair/paths/colab/n8n-workflow-repair.ipynb).

Select a T4 GPU when available, then run the notebook from top to bottom. It clones this fork and the project branch, not the organizer's upstream main branch. It uses:
- base model alias: starter (Qwen3-1.7B),
- Unsloth 4-bit base weights with LoRA adapters,
- the recipe's pinned configuration,
- the same 50 held-out prompts, retrieved official-doc excerpts and generation settings for the base and fine-tuned model,
- per-category scoring and safety checks,
- adapter/evidence saving and GGUF Q4_K_M export.

The base outputs are generated **before training**; the fine-tuned outputs are generated after training. The comparison is therefore paired on identical held-out cases.

## Evaluation

The notebook reports JSON validity, schema validity, category and diagnosis accuracy, gold-patch match, safe-abstention accuracy, patch applicability, and safety violations.

Gold-patch match is intentionally strict: a different but valid repair may be marked incorrect. Patch applicability is only a structural fixture check. Neither metric is a live n8n execution test.

## License and provenance

All examples in this recipe are original synthetic cases created for this project and released under the repository's MIT license. They are not copied from customer workflows or production exports.
