# Validation

The tests check data preparation, training mechanics, checkpoints, local inference
and browser integration. Small-model tests verify the pipeline; evaluate your
trained model separately on tasks representative of its intended use.

## Run the checks

Set up the environment using [Getting started](../GETTING_STARTED.md), including
PyTorch and both core and vision dependencies.

Run the small-model integration checks:

```sh
python test_llama_text_cycle.py
python test_llama_vision_cycle.py
python test_direct_llama_backend.py
```

For the complete suite, install the corpus-tool dependencies as well:

```sh
python -m pip install -r requirements-corpus.txt
python -m unittest discover -s tests -v
```

For browser access alone:

```sh
python -m playwright install chromium firefox webkit
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

The browser test defaults to Chromium, Firefox and WebKit. Select other installed
browsers with `CHAMP_LLM_TEST_BROWSERS`; see
[Browser setup](../VISION_AND_BROWSER.md#check-browser-access) for shell examples.
Browser launch failures fail the test. Some model tests skip when dependencies
are missing; inspect the summary and resolve missing dependencies before relying
on the result.

## Coverage

| Area | Checks |
| --- | --- |
| Model configuration | Preset counts, custom dimension validation, creation and overwrite protection |
| Text and conversations | Small-Llama weight updates, pretraining and conversation fine-tuning |
| Checkpoints | Model, optimizer, random-number state and dataset position on resume |
| Vision | Small-model image training, checkpoints and annotation integrity |
| Local inference | Model loading, text generation and image processing |
| Browser | Page loading, interactions, overflow correction and responsive captures |
| Data | Sharding, filtering, partition separation and file integrity |

The browser tests use real engines and a temporary HTTP server with predefined
model responses. They check the integration independently of a trained coding
model. Fixtures and generated test artifacts use temporary directories.

## Recorded reference run

A reference run on **2026-10-25** passed **78 tests on Windows**, including
Chromium, Google Chrome, Firefox and WebKit, with no failures or skips.
This is a historical result; rerun the checks for your checkout and environment.

| Component | Version |
| --- | --- |
| Python | 3.13.17 |
| PyTorch | 2.14.1+cpu |
| Transformers | 5.20.2 |
| Playwright | 1.64.1 |
| Chromium | 156.0.8078.4 |
| Google Chrome | 155.0.8059.84 |
| Firefox | 157.0 |
| WebKit | 27.2 |

## Validation limits

The recorded run covers small CPU models and browser execution on Windows.
It does not validate browser execution on Linux or macOS, native Safari behavior,
large-model GPU training, or Unsloth full training of custom models.

Before a long training run, verify forward/backward updates, saving and resumption
on your target hardware using [GPU training](../TRAINING_ON_RUNPOD.md).
Measure language fluency, code correctness and design quality with held-out tasks;
a passing software test suite does not establish those learned abilities.
