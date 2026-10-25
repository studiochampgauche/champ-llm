# Validation scope

The latest recorded complete run passed **78 tests on Windows with Python 3.13.17**,
including real Chromium, Google Chrome, Firefox and WebKit integration, with no failures or skips. This is evidence about the software pipeline and
small test models, not a benchmark of a trained large model.

## What the tests cover

| Area | Checks |
| --- | --- |
| Model sizes | No-dependency planning, custom dimension validation, all preset counts versus Llama tensor shapes, real tiny-model creation/training/resumption/inference |
| Text | Actual tiny-Llama weight updates, pretraining followed by conversation SFT |
| Checkpoints | Model, optimizer, RNG and streaming cursor resumption, including dropout and accumulation |
| Vision pipeline | Small-model image training, checkpoint handling and annotation integrity |
| Direct backend | Local model loading, text generation and image processing |
| Browser | Four browsers on Windows: HTTP loading, fill/press/click interactions, overflow correction, three viewports, sections and separate screenshot/annotation artifacts |
| Data | Sharding, source filtering, train/validation separation and Windows line endings |

The browser test uses real browser engines and a temporary HTTP server. Only the coding
model's answers are simulated. Test data and model artifacts are created in temporary
directories; they are not added to the training corpus.

## Reproduce

After installing dependencies as described in [Getting started](../GETTING_STARTED.md):

```sh
python test_llama_text_cycle.py
python test_llama_vision_cycle.py
python test_direct_llama_backend.py
python -m unittest discover -s tests -v
```

For browser access alone:

```sh
python -m playwright install chromium firefox webkit
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

The default browser test covers Chromium, Firefox and WebKit. This recorded run
also included installed Google Chrome, with the Safari alias resolving to WebKit:

```powershell
$env:CHAMP_LLM_TEST_BROWSERS = "chromium,chrome,firefox,safari"
python -m unittest discover -s tests -v
```

For Linux/macOS shell syntax and focused browser selection, see
[the browser guide](../VISION_AND_BROWSER.md). Selected browsers are required;
launch failures are not skipped and do not fall back to another engine.
Google Chrome is launched with the official `chrome` channel. WebKit is not the
installed Safari application, and real Safari compatibility remains a separate check.

The browser test fails when browser installation or access is unavailable.
Some model tests have optional-dependency skip guards; use the complete environment
and inspect the summary rather than treating a skipped test as a pass.

Test expectations compare resolved paths so Windows short-name aliases and full
paths do not produce false mismatches. No temporary-path normalization wrapper is
needed for the standard test command.

## Model-size validation

All eight presets are checked against real Llama parameter shapes on the PyTorch
`meta` device, without allocating billion-parameter weights. Counts are checked
with two vocabulary sizes. Additional counting checks cover tied embeddings,
attention/MLP biases and explicit head dimensions for the training size guard.

Two tiny custom architectures (different widths and layer counts) are created
with real BF16 weights, trained on CPU, resumed from checkpoints and loaded through
the direct inference backend. Tests also verify explicit selection, malformed
configuration rejection, unchanged source tokenizer files and overwrite protection.
These checks do not establish the training quality or GPU fit of the large presets.

## Reference environment

Recorded on 2026-10-25:

| Component | Version |
| --- | --- |
| Python | 3.13.17 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.20.2 |
| Playwright | 1.64.1 |
| Playwright Chromium build | 1248 |
| chromium | 156.0.8078.4 |
| chrome | 155.0.8059.84 |
| firefox | 157.0 |
| webkit | 27.2 |

These describe one tested environment, not a claim that every combination allowed
by the dependency ranges has been validated. Private machine paths and raw logs
are intentionally omitted.

## Unsloth and Python 3.13.17

Checked on 2026-10-25: [Unsloth's published package metadata](https://pypi.org/project/unsloth/)
declares Python >=3.9,<3.15, which includes 3.13.17. Its
[installation guide](https://unsloth.ai/docs/get-started/install/pip-install)
also explicitly supports Python 3.13. The final number is a patch release within
the 3.13 series, not a different minor Python version.

This establishes declared Python compatibility. The 78-test run above exercised
the project's PyTorch and browser paths; it did not validate Unsloth training.
Use an environment satisfying Unsloth's own dependency constraints and run the
[GPU training checks](../TRAINING_ON_RUNPOD.md) before a long training run.

## Still to validate

- Browser execution on Linux and macOS, and actual Safari validation. The test uses Playwright's portable
  installation mechanism; only Windows was executed for this record.
- Complete GPU training and checkpoint resumption at billion-parameter scale.
- Unsloth full training with the project's custom randomly initialized model.
- Learned multilingual fluency, reliable SaaS development and visual judgment.
- Long-context project work and temporal understanding of animations.

The current vision encoder operates on still images. The development loop does not
train model weights while it runs. Those capabilities require separate datasets,
training and evaluation.
