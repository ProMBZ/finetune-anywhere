# Work and testing report

Updated: 27 November 2026. Ready to share as a hackathon starter repository, with Colab and the JSON extractor as the demonstrated first path.

## Work completed

- Built six hardware paths, seven learning guides, three training notebooks, four task recipes, model-sharing instructions and a community showcase with validation workflows.
- Selected Apache-2.0 Qwen3 text models in 0.6B, 1.7B and 4B sizes; reviewed small Qwen3.5 candidates. Centralized model settings and updated dependent instructions.
- Rewrote Colab and Kaggle as visible install, load, LoRA, template, data, train, evaluate, save and GGUF-export cells. Added own-data instructions and downloadable run evidence.
- Authored 240 original MIT-licensed synthetic examples: 60 per recipe, split 40 training / 10 validation / 10 evaluation.
- Fixed discovery of GGUF files written to Unsloth's sibling export directory, including downloads and regression tests.
- Added all ten real before/after outputs, unchanged source evidence, checksums, measured results and remaining errors. Added a participant starting point to the main README.

## Testing completed

| Check | Result |
| --- | --- |
| Local automated suite | 41 tests passed |
| Python lint | Passed |
| Repository checks | Local links, model references, file sizes, writing rules and notebook structure passed |
| Notebook hygiene | All three notebooks have no saved execution outputs |
| Dataset/tokenizer checks | Previously verified 240 examples against three Qwen3 tokenizers: 720 length/template checks |
| Returned run evidence | All three dataset hashes and all ten evaluation prompts/references match repository data |
| Colab execution | Maintainer completed training, evaluation and saving on Tesla T4; logs show successful Q4_K_M conversion and screenshot confirms file discovery |

## Real T4 results

| Metric | Base | Fine-tuned |
| --- | --- | --- |
| Exact JSON matches | 2/10 | 7/10 |
| Directly parseable JSON | 8/10 | 10/10 |
| Valid task schema | 7/10 | 9/10 |

Training: **80.14 seconds**, three epochs, 30 optimizer steps. Peak PyTorch allocated GPU memory: **4.51 GB (4.20 GiB)**. [Full results and every response](recipes/json-extractor/results/2026-11-21-t4/).

## Validation scope

These scores describe ten held-out synthetic examples. Exported GGUF inference, Kaggle and the other hardware paths still need complete execution checks. The original ZIP predates export recovery and retains its original export status; subsequent logs and screenshot document conversion and discovery. See [maintainer notes](MAINTAINER_NOTES.md#recorded-t4-run) for provenance and remaining validation.

## Participant route

[Open Colab instructions](paths/colab/) → run the default JSON extractor → download evidence and GGUF → test locally → train on your own data → [submit your project](showcase/qwen-local-hackathon-2026/).

## Qwen n8n workflow repair v0.7 candidate

The feature branch preserves the v0.4 source corpus and original 192-case held-out evaluation while adding 348 synthetic v0.7 training cases across the eight diagnosis categories and 24 separate validation-only mapping/item-linking cases. Combined v0.5-v0.7 supplements contribute 888 authored training examples. The preparation pipeline balances 48 codes to 64 examples each (3,072 training rows), uses 156 validation rows and keeps the separate 48-case challenge set held out. This is a training candidate, not a validated model release: no v0.7 GPU run or quality score is claimed before the fresh Colab run and saved evidence.
