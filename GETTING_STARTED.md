# Getting started

Use Python 3.13 for the reference setup. Run commands from the project root.
The repository contains source code and tests. Training data, tokenizers and model
weights are created locally.

## 1. Create an environment

### Windows / PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux or macOS

```sh
python3.13 -m venv .venv
source .venv/bin/activate
```

Once activated, all commands below use `python`:

```sh
python --version
python -m pip install --upgrade pip
```

The version command should report Python 3.13.x. The repository's
`.python-version` also selects the 3.13 series for tools that read it.
If the selected interpreter is missing, install Python 3.13 for your operating
system before creating the environment.

Create a new environment on each computer instead of copying a virtual environment
between systems. Keep Windows checkout and temporary paths short if long-path
support is unavailable.

## 2. Choose a model size and a first experiment

You can preview the architecture before preparing any corpus or tokenizer:

```sh
python create_model.py --size 7b --plan
```

Replace `7b` with `20m`, `125m`, `350m`, `1b`, `3b`, `4b` or `13b` as needed.
This first preview assumes a 32,768-token vocabulary and creates no files. It does
not require ML packages. The size is always explicit; there is no default 4B model.
For custom dimensions and memory estimates, see [Model sizes](docs/MODEL_SIZES.md).

### Browser integration only — no model or GPU required

```sh
python -m pip install -r requirements-vision.txt
python -m playwright install chromium firefox webkit
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

On supported Linux distributions, missing browser libraries can be installed with
`python -m playwright install --with-deps chromium firefox webkit`. See
[the browser guide](VISION_AND_BROWSER.md) for executable overrides and the supported
operating-system requirements.

The browser test covers Chromium, Firefox and WebKit by default. Google Chrome
can be added when installed; see the browser guide for the test selection variable.

### Tiny model tests and the complete suite

Install PyTorch using its [official installation selector](https://pytorch.org/get-started/locally/)
for your operating system and hardware. A CPU build is sufficient for the tiny-model
validation tests. Then install the remaining packages:

```sh
python -m pip install -r requirements.txt -r requirements-vision.txt
python -m playwright install chromium firefox webkit
python test_llama_text_cycle.py
python test_llama_vision_cycle.py
python test_direct_llama_backend.py
python -m unittest discover -s tests -v
```

The three entry points exercise training, resumption and direct inference on
temporary tiny models. They do not create or train your chosen production model.
The complete suite also exercises the real browser; browser failures are not skipped.

For the latest recorded environment, scope and limitations, see
[Validation](docs/VALIDATION.md).

## 3. Create your own corpus

Git does not preserve empty directories. Create the local input directory:

```sh
python -c "from pathlib import Path; Path('training_data').mkdir(exist_ok=True)"
```

Add reviewed text, source code and conversation JSONL files under `training_data/`.
French, English, Spanish and other languages can share the corpus. A root-level
`training_data.txt` is an optional additional input.

Both inputs are excluded from Git by default. Keep evaluation data separate from
training examples. Review data provenance and remove credentials or personal
information before preparation. See [Datasets and languages](DATASETS_AND_LANGUAGES.md).

Documentation examples and temporary test fixtures are not training inputs.
Preparation on an empty corpus stops with an explicit error.

## 4. Prepare shards and finalize the tokenizer

```sh
python prepare_data.py --max-mb 100 --chunk-chars 8192 --shard-mb 32
python prepare_tokenizer.py --vocab-size 32768
python create_model.py --size 7b --plan
```

Preparation writes manifests and shards under `outputs/`. The tokenizer is stored
under `model/`. The plan prints an estimate without allocating model weights.
The initial tokenizer corpus should represent the languages and code you intend
to teach.

## 5. Create weights only when ready

After reviewing the plan with your finalized tokenizer, use the same size choice:

```sh
python create_model.py --size 7b --create
python verify_model.py
```

This allocates and saves the chosen randomly initialized parameters. A large model
requires substantial memory and disk space; creation does not teach it anything.
The plan reports weights and training tensors separately, excluding activations
and runtime overhead. Existing model directories cannot be overwritten.

A smaller experimental architecture is available after preparing the tokenizer:

```sh
python create_model.py --size 20m --directory models/20m --plan
python create_model.py --size 20m --directory models/20m --create
```

The `20m` profile has approximately 20 million parameters with a 32,768-token vocabulary.
The finalized tokenizer in `model/` is reused. Use `--tokenizer PATH` to select a
different local tokenizer. Alternative outputs can live under ignored `models/`;
pass their path with `--model` when training and `--local-model` in the coding agent.

## 6. Validate GPU training before scaling up

Follow [GPU training on RunPod](TRAINING_ON_RUNPOD.md) for initial GPU checks.
The current training paths use CUDA or explicitly enabled small CPU experiments.
Apple MPS training is not implemented. Browser portability does not imply GPU
training support on every device.

Run short training and checkpoint-resumption checks before extending a run.
Large-model GPU training and Unsloth integration remain unvalidated at that scale.

## 7. Extend a dataset without replacing its vocabulary

Later datasets can be added and prepared again. Keep the tokenizer fixed after
model creation. A checkpoint's saved streaming cursor depends on the exact dataset
snapshot; preserve its manifest and shards for an exact resume.

Weights, optimizer checkpoints, data and generated outputs stay outside the public
source repository. Transfer those artifacts separately when moving a training run.
