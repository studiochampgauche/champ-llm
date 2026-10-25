# Model configuration

Choose a model size that fits your goals, dataset and hardware. Use a preset for
a quick start or define custom architecture dimensions in JSON. The parameter
count follows from those dimensions and the tokenizer's vocabulary.

champ-llm creates Llama-style decoders with randomly initialized weights.
Training determines the languages and domains the model learns.

## Preview a model

```sh
python create_model.py --size 7b --plan
python create_model.py --size 1b --plan --vocab-size 48000
```

Planning reports parameter counts and memory estimates without allocating weights
or writing files. Before a tokenizer exists, it uses a provisional vocabulary of
32,768 tokens or the supplied `--vocab-size`. Omit that option to use an existing
local tokenizer's vocabulary.

Recheck the plan after finalizing the tokenizer.

## Presets and memory estimates

Preset names are approximate. These counts assume 32,768 tokens; changing the
vocabulary changes the parameter count while keeping the preset's layers unchanged.

| Preset | Text parameters | BF16 weights | FP32 training tensors |
| --- | ---: | ---: | ---: |
| `20m` | 19,925,248 | 0.04 GB | 0.32 GB |
| `125m` | 128,206,080 | 0.26 GB | 2.05 GB |
| `350m` | 350,274,560 | 0.70 GB | 5.60 GB |
| `1b` | 1,020,348,416 | 2.04 GB | 16.33 GB |
| `3b` | 2,989,309,440 | 5.98 GB | 47.83 GB |
| `4b` | 4,026,768,384 | 8.05 GB | 64.43 GB |
| `7b` | 7,002,705,920 | 14.01 GB | 112.04 GB |
| `13b` | 12,997,575,680 | 26.00 GB | 207.96 GB |

BF16 weights use 2 bytes per parameter. The FP32 column covers weights, gradients
and two AdamW states at 16 bytes per parameter. Both columns exclude activations,
temporary tensors and runtime overhead, so they are not total memory requirements.
Vision parameters are additional, and Unsloth's precision and optimizer choices
can change memory use.

Model creation allocates the entire model in memory. The training scripts use a
single device and do not distribute a model across GPUs. Check available RAM,
VRAM and disk space, then validate a short run before committing to longer training.

## Create the model

Prepare your corpus and tokenizer using [Getting started](../GETTING_STARTED.md),
then review and create the selected architecture:

```sh
python create_model.py --size 7b --plan
python create_model.py --size 7b --create
```

Select exactly one of `--size` or `--config` with either action. Creation uses the
finalized tokenizer; `--vocab-size` is only for provisional plans.
The output includes `model_identity.json` with the model's dimensions and count.

For a small experiment in a separate directory:

```sh
python create_model.py --size 20m --tokenizer model --directory models/first-experiment --plan
python create_model.py --size 20m --tokenizer model --directory models/first-experiment --create
python train_pytorch.py --model models/first-experiment --output training_runs/first-experiment --steps 1 --allow-cpu
```

CPU training is limited to models with at most 100 million parameters.

## Custom dimensions

Copy and edit [the architecture example](../configs/model-small.example.json), then
pass your configuration:

```sh
python create_model.py --config configs/model-small.example.json --plan
python create_model.py --config configs/model-small.example.json --create
```

The JSON schema requires five positive integers:

| Field | Meaning |
| --- | --- |
| `hidden_size` | Width of the decoder |
| `intermediate_size` | Width of the feed-forward layers |
| `num_hidden_layers` | Number of decoder blocks |
| `num_attention_heads` | Number of query attention heads |
| `num_key_value_heads` | Number of key/value attention heads |

`max_position_embeddings` is optional and defaults to 4096. The hidden size must
be divisible by the attention head count, the attention head count must be divisible
by the key/value head count, and each head dimension must be even.

Custom dimensions let you choose sizes beyond the supplied presets, subject to
these architecture rules and available resources. The resulting parameter count
also depends on vocabulary size. The architecture uses separate input/output
embeddings, bias-free attention and feed-forward layers, SiLU and rotary embeddings.

## Keep experiments separate

To try another size, use a new model directory and training run. `--create`
protects existing weights, and a separate output directory must be empty.
The source tokenizer is copied without changing its token IDs.

A new architecture needs new weights and training. Resuming a checkpoint uses
its saved architecture and training state. A vision adapter must match the language
model's hidden dimension.

`model/`, `models/` and `training_runs/` are ignored by Git. Add equivalent ignore
rules if you store artifacts elsewhere. See [Validation](VALIDATION.md) for model
creation and training test coverage.
