# Conversations and training checkpoints

Conversation training teaches a pretrained base model to respond to instructions
and dialogue. Prepare a representative text/code corpus first, then use reviewed
conversations for supervised fine-tuning.

## Conversation format

Provide JSONL records with a `messages` array containing `role` and `content`.
Content may use any language in your curriculum. Training learns assistant
responses; `"learn": false` excludes a particular assistant response from the loss.

See [Datasets and languages](DATASETS_AND_LANGUAGES.md#conversation-format) for an
example. Keep the tokenizer's role and end-of-message markers unchanged.

## Train

Start with a short pretraining check:

```sh
python train_pytorch.py --mode pretrain --steps 1 --context 128
```

Once your base model has completed its intended pretraining, pass its saved model
directory to the conversation trainer:

```sh
python train_conversations.py --model /path/to/trained/model --steps 1 --context 128
```

These commands run one step to check the pipeline. Choose a training duration and
evaluate the model before using it for real tasks. See [GPU training](TRAINING_ON_RUNPOD.md)
for hardware setup and Unsloth checks.

## Save and resume

Checkpoints are stored under
`training_runs/<engine>_<mode>/checkpoints/step-NNNNNNNN/`. Each contains model
weights, tokenizer, optimizer state, random-number state, dataset position and
training metadata. The run's `latest.json` points to the newest complete checkpoint.

Use `--resume` with the same run and compatible settings. For text training,
`--steps` is the total step target, including completed steps. Keep the dataset
manifest and referenced shards unchanged for an exact resume.

Back up complete checkpoints when moving a run between machines. Saving the model
weights and tokenizer supports text inference, but does not preserve everything
needed to continue the same training run.

## Use corrections in later training

The [coding agent](DIRECT_AGENT.md) can inspect a website, propose changes and run
your tests. Review successful corrections before adding them to a dataset.
They become part of the model's learning only when you run another training stage.
