# Maintainer notes

## Qwen n8n repair dataset v0.7

Prepared as a new training candidate on 27 November 2026. The original v0.4 source corpus (640 train / 128 validation / 192 held-out evaluation) remains intact. v0.5/v0.6 contributed 540 supplemental synthetic training cases; v0.7 adds 348 more across all eight categories (84 mapping, 72 item linking, 48 node configuration, 36 Code node, 36 execution/API, 24 expression, 24 workflow structure and 24 credential-safety cases). The v0.7 validation-only supplement adds 24 mapping/linking cases. The pipeline balances the 1,528 authored synthetic training scenarios to 64 examples per code (3,072 total), creating 1,544 controlled fill variants. Those variants are not independent incidents.

Prepared validation has 156 cases (128 original + 4 v0.6 + 24 v0.7). The original 192-case evaluation and separate 48-case challenge evaluation remain outside training. QLoRA remains rank 16 / alpha 32, with learning rate 1e-5, two epochs, 2,048-token context and effective batch size 8. Validation/checkpoint saves run every 128 steps (768 planned optimizer steps); best checkpoint selection uses validation loss. GGUF export/download remain disabled.

**Validation boundary:** dataset checks, patch-fixture validation, tests and CI must pass before GPU training. No v0.7 model-quality result exists until the fresh Colab run finishes. The 90% goal is not guaranteed; the legacy benchmark has four cases per diagnosis code and synthetic fixtures do not prove live n8n execution.

## Validation status

Updated 21 November 2026. The maintainer supplied the Colab evidence ZIP, successful Q4_K_M conversion logs and a screenshot confirming artifact discovery and download initiation. The original notebook searched the wrong directory after conversion; commit 50b8b09 fixes discovery in both notebooks. The returned evidence is preserved unchanged and independently scored. See the recorded T4 run below.

| Area | Evidence | Remaining check |
| --- | --- | --- |
| Models | Official Hugging Face cards/API and Unsloth catalog checked for all six reviewed small models | Review model revisions when updating the registry |
| Recipe data | 60 independently authored scenarios per recipe; 40 train, 10 validation, 10 evaluation; original MIT data | Domain review and broader real-world evaluation |
| Cloud notebooks | Explicit loading, LoRA, template, data, training, inference, saving and export cells; install sequence checked against current official Unsloth notebooks | Fresh run of the corrected notebook; Kaggle execution |
| Training environments | Existing Linux pins retained; Mac CLI environment installed during initial build | Hardware training, adapter fusion and quantized inference for the new Qwen3 defaults |
| Alternative tools | Qwen3 reasoning-template settings and merge/export handling updated | End-to-end runs on supported GPUs |
| Public results | [JSON extractor T4 run](recipes/json-extractor/results/2026-11-21-t4/), all ten before/after outputs | Quantized inference and broader evaluation |

The initial repository's CI passed with 25 tests. The export fix passes 41 local tests plus notebook structure/output checks. All 240 examples also pass exact token-length and train/inference template equivalence checks with each of the three actual Qwen3 tokenizers (720 example checks). Dependency resolution, API inspection and tests do not substitute for a complete GPU run. The earlier Mac attempt stopped before training because the filesystem had less than 1 GB free; it provides no model-quality evidence.

## Recorded T4 run

[The original evidence and all outputs](recipes/json-extractor/results/2026-11-21-t4/) show exact matches increasing from 2/10 to 7/10. All split hashes and evaluation prompts/references match repository data. Training took 80.14 seconds; peak PyTorch allocated memory was 4.51 GB (4.20 GiB), not total device usage. Installation, downloads, inference and export are excluded from that duration. The ten synthetic cases are a small teaching evaluation, not evidence of general reliability. Three tuned answers still fail exact match; item boundaries in these references merit domain review. References have not been changed after seeing outputs.

The comparisons evaluate the four-bit base with the adapter disabled versus enabled, before GGUF export. The loaded revision differs from the reviewed catalog revision; both are preserved in the original manifest. This records what actually ran rather than asserting revision identity or deterministic reproduction from today's unpinned install.

The ZIP contains `export_completed: false` because its report predates recovery. It has not been rewritten. Subsequent supplied logs report successful Q4_K_M conversion, and the screenshot shows recovery finding the nonempty file in `gguf_gguf` and initiating download. These establish conversion and discovery, not a completed local download or successful GGUF inference. Export time and GGUF checksum are unavailable. To refresh the ZIP in the same session, rerun section 15 after the recovery cell; no retraining is needed.

## Model review

The registry is the only source of model IDs, including the Unsloth catalog counterparts. License verification uses the official Qwen cards; the Qwen3 Unsloth API metadata references those base models but does not repeat a license field. Official cards report rounded total sizes of 0.6B, 1.7B and 4B for the selected Qwen3 models. They are text-generation models under Apache 2.0. The small Qwen3.5 cards and Unsloth listings were checked too: 0.8B, 2B and 4B are Apache 2.0 image-text models. They are recorded as reviewed candidates, not selectable aliases.

Decision: use the text-only Qwen3 family across all existing paths. Unsloth publishes Qwen3.5 vision notebooks, but that is not evidence of identical text-only training, MLX fusion and llama.cpp export behavior for the multimodal weights. No cross-tool multimodal compatibility claim is made. The older non-Apache larger model was removed.

The memory table uses the rounded parameter totals explicitly stated in model cards. Serialized tensor metadata can count stored tensors differently; these estimates are not downloaded file sizes or measured VRAM. Registry revisions record what was reviewed. Run manifests separately record the revision actually loaded by the trainer.

The saved Qwen3 tokenizer prepends a Jinja assignment fixing `enable_thinking` to false. The original role delimiters and empty thinking prefix are preserved. Unsloth, MLX and TRL save this template with their outputs. LLaMA-Factory uses the Qwen3 reasoning template with thinking disabled; SWIFT adds the non-thinking prefix during training; Axolotl receives the configured tokenizer's Jinja template. The shared adapter merger writes the same configured tokenizer.

## Cloud notebook installation

The Colab notebook follows the current official Qwen3 reasoning/conversation notebook: install the Colab helper packages, install Unsloth's stack without replacing the whole host dependency tree, install kernels, then select Transformers 4.57.6 and TRL 0.22.2. Kaggle follows Unsloth's platform notebook with its CUDA wheel setup. Both keep install logs visible and add PyYAML for the registry. Neither installs the 100-line Linux requirements file.

The returned Colab T4 run records a successful training execution with the actual package versions. Kaggle execution remains unverified. Upstream unpinned packages can change. The evidence report captures actual versions, loaded revision, dataset hashes, settings, timings, peak allocated GPU memory and export outcome. A fresh session is needed after replacing already-imported libraries.

The notebooks record base answers with the adapter disabled, then train and evaluate the same held-out prompts. Targets never enter the inference prompt. Before/after results are for the shared four-bit base plus adapter, before GGUF export. Exported GGUF quality is a separate evaluation. The small report ZIP excludes weights and credentials.

## Data authorship and splits

All 240 scenarios were written individually for this repository. There is no identifier substitution or combinatorial expansion. JSONL serialization and unified-diff formatting were performed programmatically; scenario content and targets were supplied explicitly. The data is synthetic, original and MIT-licensed, not scraped user records or copied repository code. Each data/manifest.json records provenance and split counts.

The new datasets deliberately trade repetitive volume for broader coverage. Each has 40 training, 10 validation and 10 evaluation scenarios. Training uses three epochs. The SQL examples span distinct schemas, and support examples supply distinct policies. Templates such as chat roles and SQL syntax naturally recur, but cases are not generated by changing product or table numbers. These remain small teaching datasets, not production benchmarks; add domain-specific examples and independently sourced evaluation data.

## Decisions where the spec was silent

- Use Python 3.11 across training and contributor tooling. Utilities use the standard library except YAML/schema handling and optional tokenizer-based counting.
- Use Apache-2.0 text-only Qwen3 models. Exact IDs, catalog counterparts and reviewed revisions live in the registry.
- All guide commands start from the repository root. Windows GPU training uses WSL2. Keep each trainer in its own environment.
- Dataset size is 60 authored scenarios per recipe, split 40/10/10. See the data authorship section above.
- Add a validation split separate from the final evaluation set. The trainer never loads evaluation data for fitting or hyperparameter selection.
- Training uses full-conversation loss. Assistant-only loss is not enabled without verifying model-specific template masks.
- MLX defaults to eight adapter layers and 100 iterations. Scale 2 with rank 16 corresponds to the alpha/rank ratio used by the GPU recipe, but iterations, layers and optimizer behavior do not make the runs equivalent.
- MLX fusion reloads the full-precision base instead of fusing and dequantizing a lossy base. Conversion to GGUF goes through llama.cpp because the built-in MLX exporter has architecture restrictions.
- NVIDIA run manifests separate training, preparation and export durations, record GPU/package information and hash all dataset splits. The manifest is saved before export so a conversion failure does not erase training evidence.
- CPU float32 merging is portable but may need substantial RAM. It is a separate operation from low-memory training.
- Memory numbers are decimal GB. The calculator exposes KV architecture assumptions. Its training budget is a labelled heuristic, not a measured architecture-specific activation calculation.
- compare.py uses local OpenAI-compatible servers so it works across hardware without installing two model stacks. It saves actual text and raw JSON. Unit tests use a clearly isolated mock HTTP server, not a published model example.
- Published JSON extractor outputs come from the supplied T4 evidence. A hero GIF still requires a real local inference recording.
- Model IDs are centralized for first-party workflows. User showcase entries, externally uploaded model cards and the preserved original T4 run manifest must identify their actual base model and are exceptions to the registry-only rule.
- The required submission template is intentionally a blank form. It is skipped by gallery discovery and fails validation until completed. There are no fictional gallery entries.
- Future event names are accepted only with an existing event README and a matching folder. Enum restrictions apply to the training tool, platform, method and runtime fields.
- The deadline's closing minute is inclusive. Late means PR creation at or after 2026-11-28T19:00:00Z. Edits do not change the creation timestamp.
- Newest gallery entries are ordered by their most recent manifest commit timestamp, then path for deterministic ties. Uncommitted test entries use timestamp zero. No participant tracking is added.
- Internal Markdown links are blocking checks. External links are checked in CI but transient network/auth failures are warnings. Notebook badges include a manual import fallback where needed.
- The trusted showcase validator comes from the PR base commit. Submission files are data only. The write-enabled comment workflow never checks out code, constrains archive extraction and text lengths, validates PR/run association and derives labels from GitHub metadata rather than artifact claims.
- Automatic table updates use the GitHub Actions bot identity in CI only. Local commits use the maintainer's default Git account. Main-branch rules must permit the intended bot update strategy or the workflow needs to open update PRs instead.
- CODE_OF_CONDUCT.md uses Contributor Covenant 2.1 with the maintainer's supplied account email for private reports and punctuation normalized to this repository's writing rule.

## Verified sources

- [Official Unsloth Qwen3 Colab notebook](https://github.com/unslothai/notebooks/blob/main/nb/Qwen3_%2814B%29-Reasoning-Conversational.ipynb).
- [Official Unsloth Qwen3 Kaggle notebook](https://github.com/unslothai/notebooks/blob/main/nb/Kaggle-Qwen3_%284B%29-Instruct.ipynb).
- [Unsloth model catalog](https://huggingface.co/unsloth) and [official Qwen models](https://huggingface.co/Qwen).
- [SWIFT reasoning parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md) and [LLaMA-Factory templates](https://github.com/hiyouga/LLaMA-Factory/blob/main/src/llamafactory/data/template.py).

Model cards and API metadata were fetched directly from the Hugging Face repositories named in configs/models.yaml. All selected and reviewed model licenses are Apache 2.0. Catalog counterparts were verified under the Unsloth organization.

- [Unsloth source and installation](https://github.com/unslothai/unsloth), with package metadata checked for the pinned release.
- [MLX LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md), plus installed CLI help and adapter configuration source.
- [LLaMA-Factory training example](https://github.com/hiyouga/LLaMA-Factory/blob/main/examples/train_lora/qwen3_lora_sft.yaml) and [dataset mappings](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md).
- [ms-swift quick start](https://github.com/modelscope/ms-swift), including current tuner flags and custom data references.
- [Axolotl conversation data](https://github.com/axolotl-ai-cloud/axolotl/blob/main/docs/dataset-formats/conversation.qmd) and [QLoRA configuration](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3/32b-qlora.yaml).
- [TRL SFTConfig source](https://github.com/huggingface/trl/blob/v1.15.0/trl/trainer/sft_config.py) and [SFTTrainer source](https://github.com/huggingface/trl/blob/v1.15.0/trl/trainer/sft_trainer.py).
- [llama.cpp converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py) and [Ollama import documentation](https://docs.ollama.com/import).
- [ModelScope SDK](https://github.com/modelscope/modelscope); installed CLI and Python signatures checked for version 1.40.3.
- [Colab FAQ](https://research.google.com/colaboratory/faq.html) and [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks). No fixed quota is stated.
- Provider references are linked in the cloud guide. No prices or guaranteed availability are claimed.

## Unverified or deliberately omitted

- AMD training folder: omitted because no end-to-end ROCm training setup was validated on available hardware. This does not mean AMD cannot train models.
- Full Modal deployment: the architecture is described with the official GPU guide, but an untested provider-specific app is not presented as working code.
- The maintainer ran Colab training and export; exported GGUF inference and a fresh run of the corrected notebook remain unverified.
- AWQ/GPTQ commands are omitted because hardware, model and serving-version compatibility must be checked for a specific deployment.
- Guaranteed times, free GPU quotas, costs and claims of universal compatibility are omitted. Measured training duration is reported only for the supplied run.

## Real-run backlog

Run each recipe on at least one complete training-to-local-inference path and add only actual outputs:

- Run commit-message-writer and add real before/after outputs, elapsed time and failures.
- Completed: JSON extractor training outputs, field-level accuracy and parseability counts. Next: evaluate its exported GGUF.
- Run sql-generator and add real outputs with SQLite result-equivalence checks.
- Run support-bot-from-docs and add grounded-answer and abstention results.
- Repeat evaluation after GGUF or MLX quantization. Include hardware, software versions, revision and context length.
- Record a terminal GIF of a fine-tuned model running in Ollama.

## Repository administration

- Completed: Discussions enabled; Q&A, Show and tell, and Ideas categories are present.
- Completed: repository is public under the default maintainer account and its description is set.
- Completed topics: fine-tuning, llm, qwen, lora, qlora, unsloth, mlx, llama-factory, gguf, ollama, llama-cpp, local-llm, colab, kaggle, hackathon.
- Upload a social preview image. No fabricated demo screenshot is supplied.
- Install the all-contributors app and add people manually after merging.
- Protect main and require repository checks. Require showcase validation for showcase submissions. A path-filtered validate workflow cannot be a universal required check without leaving unrelated PRs pending; use conditional rules or change the workflow to always run a no-op job outside showcase changes.
- Completed: showcase, hackathon-2026, community and late labels created.
- Decide whether table-update bot pushes bypass protection or should become PRs.
- Run a real valid and a deliberately broken submission through a fork PR, verify comments update in place, and check late labels with a controlled test of the timestamp logic.
- Record the hero GIF and replace the README's event block with actual winners after results are supplied.
