# Getting started

Use Python 3.13 and run the commands below from the project root.

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

Activate the environment, then check Python and update pip:

```sh
python --version
python -m pip install --upgrade pip
```

The version should be Python 3.13.x. Install it for your operating system if needed,
and create a fresh virtual environment on each computer.

## 2. Plan your model

Preview a model's parameter count and memory estimates:

```sh
python create_model.py --size 7b --plan
```

Choose a preset or define your own dimensions with `--config`. Planning can run
before you have data, a tokenizer or ML dependencies. Without a tokenizer, it uses
a provisional 32,768-token vocabulary; rerun the plan after preparing your tokenizer.
See [Model configuration](docs/MODEL_SIZES.md) for available options and hardware
requirements.

## 3. Install dependencies and check the setup

Install PyTorch using its [official installation selector](https://pytorch.org/get-started/locally/)
for your operating system and hardware, then install the project dependencies:

```sh
python -m pip install -r requirements.txt -r requirements-vision.txt
```

A CPU build is sufficient for the small-model integration checks:

```sh
python test_llama_text_cycle.py
python test_llama_vision_cycle.py
python test_direct_llama_backend.py
```

These checks train and load temporary small models to verify the pipeline.

For website capture and the coding agent, install the browser engines:

```sh
python -m playwright install chromium firefox webkit
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

The browser check can run without a trained model or GPU. See
[Browser setup](VISION_AND_BROWSER.md) for Google Chrome, Linux system libraries
and browser selection. Contributors can find the complete test commands in
[Validation](docs/VALIDATION.md).

## 4. Prepare your corpus and tokenizer

Create the input directory:

```sh
python -c "from pathlib import Path; Path('training_data').mkdir(exist_ok=True)"
```

Add reviewed UTF-8 text, source code and conversation JSONL under `training_data/`.
A root-level `training_data.txt` is an optional additional input. Keep evaluation
examples separate. See [Datasets and languages](DATASETS_AND_LANGUAGES.md) for formats.

Prepare the data and train the tokenizer:

```sh
python prepare_data.py --max-mb 100 --chunk-chars 8192 --shard-mb 32
python prepare_tokenizer.py --vocab-size 32768
python create_model.py --size 7b --plan
```

Prepared shards and manifests are saved under `outputs/`; the tokenizer is saved
under `model/`. Its training corpus should represent the languages and code you
want the model to learn.

## 5. Create your model

Once your tokenizer is finalized and the plan fits your resources:

```sh
python create_model.py --size 7b --create
python verify_model.py
```

Creation allocates and saves randomly initialized weights. The model still needs
training. Memory estimates exclude activations and runtime overhead; see
[Model configuration](docs/MODEL_SIZES.md) before creating a large model.

For a smaller first experiment, reuse the tokenizer and choose a separate output:

```sh
python create_model.py --size 20m --tokenizer model --directory models/first-experiment --plan
python create_model.py --size 20m --tokenizer model --directory models/first-experiment --create
python train_pytorch.py --model models/first-experiment --output training_runs/first-experiment --steps 1 --allow-cpu
```

The `20m` preset has approximately 20 million parameters with a 32,768-token
vocabulary. Pass the saved directory with `--model` for training or
`--local-model` for the coding agent.

## 6. Train and evaluate

Follow [GPU training](TRAINING_ON_RUNPOD.md) to check CUDA training and checkpoint
resumption before a longer run. Explicit CPU training is available for models
with at most 100 million parameters. Apple MPS and distributed multi-GPU training
are not implemented.

After pretraining, use [conversation training](CONVERSATIONS_CHECKPOINTS_AGENT.md)
to teach instruction following. Evaluate progress on held-out examples before
expanding the training run. Large-model GPU training and Unsloth integration
remain experimental.

## 7. Continue with new data

You can add datasets and prepare them again while keeping the tokenizer fixed.
Preserve the original manifest and shards when resuming an existing checkpoint;
its saved data position depends on that snapshot.

Corpora, weights, checkpoints and generated outputs are ignored by Git. Transfer
them separately when moving a training run to another machine.
