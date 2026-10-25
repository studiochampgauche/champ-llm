# Choose your model size

`champ-llm` creates randomly initialized Llama-style decoders. Select a
preset or custom dimensions before creating weights. Model size controls capacity
and resource requirements; your datasets and training determine the domains and
languages learned. A larger model is not a guarantee of better results.

## Preview without a corpus or tokenizer

```sh
python create_model.py --size 7b --plan
python create_model.py --size 1b --plan --vocab-size 48000
```

Planning without a local tokenizer uses a provisional vocabulary of 32,768 tokens,
or the supplied `--vocab-size`. This path uses only the Python standard library and
creates no files or weights. If a local tokenizer exists, omit `--vocab-size` to
count its actual vocabulary instead. Recheck the plan after finalizing the tokenizer.

## Presets

Counts below assume 32,768 tokens and untied input/output embeddings. Preset
names are approximate, and their dimensions are fixed. Changing vocabulary size
changes the count, not the preset's number of layers. These are experimental local
configurations, not pretrained vendor models or claims about model quality.

| Preset | Text parameters | BF16 weights alone | FP32 PyTorch training tensors alone |
| --- | ---: | ---: | ---: |
| `20m` | 19,925,248 | 0.04 GB | 0.32 GB |
| `125m` | 128,206,080 | 0.26 GB | 2.05 GB |
| `350m` | 350,274,560 | 0.70 GB | 5.60 GB |
| `1b` | 1,020,348,416 | 2.04 GB | 16.33 GB |
| `3b` | 2,989,309,440 | 5.98 GB | 47.83 GB |
| `4b` | 4,026,768,384 | 8.05 GB | 64.43 GB |
| `7b` | 7,002,705,920 | 14.01 GB | 112.04 GB |
| `13b` | 12,997,575,680 | 26.00 GB | 207.96 GB |

The last column covers FP32 weights, gradients and two AdamW states (16 bytes per
parameter). Activations, temporary tensors and runtime overhead are additional.
Unsloth's optimizer and precision choices have different requirements. Neither
column is a total VRAM requirement or a guarantee that training will fit.
The optional vision encoder adds parameters that are not included in this table.
The current training scripts do not distribute a model across multiple GPUs.

## Create the chosen model

Prepare your corpus and tokenizer using [Getting started](../GETTING_STARTED.md),
then review and create the same architecture:

```sh
python create_model.py --size 7b --plan
python create_model.py --size 7b --create
```

Both `--plan` and `--create` require exactly one of `--size` or `--config`.
There is no implicit size and no automatic training. Creation uses the actual
local tokenizer; `--vocab-size` is available only for a provisional plan.
The saved `model_identity.json` records the profile, dimensions and actual count.

For a small experiment in a separate directory, reusing the finalized tokenizer:

```sh
python create_model.py --size 20m --tokenizer model --directory models/first-experiment --plan
python create_model.py --size 20m --tokenizer model --directory models/first-experiment --create
python train_pytorch.py --model models/first-experiment --output training_runs/first-experiment --steps 1 --allow-cpu
```

`models/`, `model/` and `training_runs/` are ignored by Git.
A model outside those directories needs equivalent ignore rules for its tokenizer
and metadata. The `20m` preset uses the same creation and training pipeline as
the larger presets. The repository includes no training data.

## Custom dimensions

Start from [the small architecture example](../configs/model-small.example.json):

```sh
python create_model.py --config configs/model-small.example.json --plan
python create_model.py --config configs/model-small.example.json --create
```

The JSON schema has five required positive integer fields:

- `hidden_size`: width of the decoder.
- `intermediate_size`: width of its feed-forward layers.
- `num_hidden_layers`: number of decoder blocks.
- `num_attention_heads`: query attention heads.
- `num_key_value_heads`: key/value attention heads.

`max_position_embeddings` is optional and defaults to 4096. The hidden size must
be divisible by the attention head count, the head count must be divisible by the
key/value head count, and each head dimension must be even. Unknown fields and
invalid dimensions are rejected before allocation. This is an architecture input
file, not an arbitrary Hugging Face checkpoint configuration.

The count is derived from these dimensions and the vocabulary; it is not an
independent setting. The implementation fixes untied embeddings, bias-free
attention/MLP layers, SiLU and standard rotary embeddings. Valid dimensions do not
establish training quality, hardware fit or Unsloth compatibility.

## Keep each model and training run separate

Existing model weights cannot be overwritten by `--create`. To try another size,
choose another output directory and training run. A separate output must be empty;
the source tokenizer is read and saved into the new model without changing its IDs.

Changing a size flag does not resize an existing trained model. A new architecture
needs its own weights and training. Resuming a checkpoint reloads its saved
architecture, optimizer, RNG and data cursor; it does not apply another size choice.
A vision adapter also remains tied to the language model's hidden dimension.

The presets' parameter counts are checked against actual Llama tensor shapes on
PyTorch's `meta` device without allocating large weights. Real creation, weight
updates, resumption and inference are exercised on tiny custom models. See
[Validation](VALIDATION.md) for the scope of the latest run.
