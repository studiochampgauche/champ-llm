# Conversations and training checkpoints

## Conversation format

Input JSONL has `messages` with `role` and `content`, both English schema names. The `content` may be written in French, English, or any other language. Use `"learn": false` on assistant responses you want to exclude from the loss. The tokenizer reserves `<|user|>`, `<|assistant|>`, `<|system|>`, `<|tool|>`, and `<|end|>` markers. Do not rename these markers.

The conversation loader masks the prompt roles and trains **assistant responses only**. Pretraining on code/text and supervised instruction following are different training phases. It is usually inappropriate to begin with conversational fine-tuning before the blank model has learned enough language to process instructions.

## Commands

```bash
# Pretraining, full weights (only after GPU smoke tests)
python train_pytorch.py --mode pretrain --steps 1 --context 128

# After a trained base exists, use its checkpoint model path:
python train_conversations.py --model /path/to/trained/model --steps 1 --context 128
```

The trainer saves a full checkpoint under `training_runs/<engine>_<mode>/checkpoints/step-NNNNNNNN/` with its tokenizer, model weights, optimizer, RNG state, dataset cursor and metadata. The run's `latest.json` points to the newest complete checkpoint. Resume with `--resume`, but keep all data, training settings, and checkpoint files unchanged.

Unsloth and PyTorch support paths are distinct. Compatibility of full-weight Unsloth training on custom blank model weights must be verified on a real rented GPU.

## Agent loop

The coding agent can inspect a running local website, capture screenshots, run an **explicit user-supplied** test command, request code edits from a local trained model or Ollama, and iterate. Only `--apply` writes edits; run without it first. The local model cannot serve as an effective coder before extensive training and evaluation. See [DIRECT_AGENT.md](DIRECT_AGENT.md).
