# Release notes

## Small-model naming

The smallest size preset is named `20m`, consistent with the other parameter-count
presets. Its architecture is unchanged: 19,925,248 parameters with a 32,768-token
vocabulary. Use `--size 20m` when planning or creating it. Explicit CPU training
now uses `--allow-cpu`, retaining the existing 100-million-parameter limit.

## Multiple browsers through Playwright

Capture and coding-agent commands now accept repeatable `--browser` options for
Chromium, installed Google Chrome, Firefox and WebKit. `safari` is an explicit
alias for WebKit, not the Safari application. Chromium remains the default.
Captures and audit files are separated by browser, with actual engine/channel/
version metadata. The development loop checks every selected browser before and
after edits and reviews vision inputs per browser when a vision backend is loaded.
The real browser test defaults to Chromium, Firefox and WebKit; installed Chrome
can be included with `CHAMP_LLM_TEST_BROWSERS`. Selected engines must launch;
missing browsers are not skipped or replaced with another engine.
See [the browser guide](VISION_AND_BROWSER.md) for installation and commands.

## Project name

The project is named `champ-llm`, developed by **Studio Champ Gauche**.
New model identities, agent reports, dataset descriptions and publication examples
use this name. Existing weights and datasets are not renamed or migrated.
The browser override is now `CHAMP_LLM_CHROMIUM`; previous environment names remain
accepted. Python callers may use `ChampBackend`; the previous class name remains
available as a compatibility alias.

## Selectable model sizes

The project is now named `champ-llm`. Model creation requires an explicit
`--size` preset or `--config` JSON file. Presets include `20m`, `125m`, `350m`,
`1b`, `3b`, `4b`, `7b` and `13b`; the latter names are approximate parameter counts.
`--plan` works before dataset/tokenizer preparation and allocates no weights.
Custom dimensions are validated, and creation records the actual parameter count.
See [Model sizes](docs/MODEL_SIZES.md) for the CLI and resource estimates.

Older creation commands without a size must now include one explicitly.
Existing models and training checkpoints are not resized or migrated by this change.

## English naming and interfaces

This package replaces the previous French-named `MonCerveau` distribution with `champ-llm` while keeping **French available in training data and user prompts**.

## Renamed entry points

| Old | New |
|---|---|
| `preparer_donnees.py` | `prepare_data.py` |
| `preparer_tokenizer.py` | `prepare_tokenizer.py` |
| `creer_modele.py` | `create_model.py` |
| `entrainer_unsloth.py` | `train_unsloth.py` |
| `entrainer_pytorch.py` | `train_pytorch.py` |
| `entrainer_conversations.py` | `train_conversations.py` |
| `capturer_site.py` | `capture_website.py` |
| `atelier_agent.py` | `coding_agent.py` |
| `backend_local.py` | `local_backend.py` |

Other Python modules, tests and technical guides have also been renamed. CLI flags are in English (e.g. `--steps`, `--model`, `--output`, `--resume`, `--apply`, `--goal`). JSON schema fields are English, including conversation assistant flag `learn` and vision annotation fields `project`, `code_files`, `improvements`, `answer`.

## Existing local project migration

1. **Back up** your old project, dataset, tokenizer, model and any checkpoints before changing scripts.
2. Move your own source documents from `mes_donnees/` into the new `training_data/` folder; copy the old `mes_donnees.txt` as `training_data.txt` if used.
3. Do not overwrite new Python scripts with old names. Use the new ZIP as a clean project and copy only your own datasets.
4. Replace older conversation JSONL assistant flag `"apprendre"` with `"learn"`; review negatives carefully so unwanted responses are not trained.
5. Update visual `.vision.json` annotations to the new **English** field names. Keep their existing `image_sha256` unchanged and review the image before preparing again.
6. Regenerate `outputs/` shards using `python prepare_data.py`; old shards are not guaranteed compatible with the new schema. Do not regenerate an already-used tokenizer.
7. If you already trained your old model/checkpoints, treat the new data/CLI/checkpoint schemas as **potentially incompatible**. Keep your old working environment and validate migrations on a copy instead of assuming resume will work.

## Verification limitations

Basic preparation and CPU unit tests are covered; large-model Unsloth GPU training, visual understanding and end-to-end trained agent behavior remain unverified. Use one-step smoke tests before costly runs. Developer-facing messages and generated metadata use English names. The training corpus has been cleared; multilingual examples remain only in documentation and isolated test fixtures.

## Validated fixes preserved during replacement

- Restored the CPU-only real-Llama text test, including changed-weight checks, pretraining-to-SFT continuity, dropout, gradient accumulation, and exact resumed weights/optimizer/RNG/cursor checks.
- Restored the Windows LF/CRLF preservation regression test.
- Completed English messages and remaining generated metadata names, including `model_identity.json`, `integrity_version`, `model_config`, `tokens_per_tile`, screenshot dimensions and pixel-difference reports. Existing artifacts are not silently migrated.
- Cleared all bundled training examples and the optional text input. The project is ready for new datasets.
