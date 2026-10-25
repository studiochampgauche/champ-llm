# GPU training on RunPod

Your chosen model architecture is randomly initialized. Review its size with `create_model.py --size YOUR_PRESET --plan` before allocation. **A GPU with enough memory is only the beginning**: pretraining a useful coding model from scratch requires massive datasets and substantial compute. Model quality must be measured on held-out tasks after training.

## Suggested first run: one step

1. Rent a machine with a compatible NVIDIA GPU and adequate VRAM/storage. Use a supported PyTorch/CUDA image; verify it before installing other dependencies.
2. Copy the project, prepared `outputs/` shards, `model/` weights and tokenizer into `/workspace/champ-llm/`. Use a persistent volume or off-pod backups for checkpoints.
3. In the RunPod terminal:

```bash
cd /workspace/champ-llm
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt
python -m pip install unsloth datasets trl bitsandbytes
python verify_unsloth.py --directory model
python train_unsloth.py --mode pretrain --steps 1 --context 128
```

Unsloth full training of a custom **blank** Llama checkpoint has **not been verified**. The load check is not a substitute for the one-step forward/backward/save/resume check. If Unsloth is incompatible, investigate before paying for longer training; a PyTorch fallback is provided:

```bash
python train_pytorch.py --mode pretrain --steps 1 --context 128
```

Actual memory usage may exceed advertised GPU VRAM depending on dtype, optimizer, activations, and implementation. Avoid continuing if CUDA OOM or non-finite losses occur. Saving optimizer states can take considerable disk space.

## Resuming a run

For a previously initialized output run (example with PyTorch):

```bash
python train_pytorch.py --mode pretrain --steps 2 --context 128 --resume
```

**Text steps are total steps** (`--steps 2` means finish at step 2), whereas vision training uses **additional steps**; see the separate guide. Keep `outputs/shards_manifest.json` and every referenced shard unchanged during resume. Download the complete run checkpoints, not just the model weights, if you need exact optimizer/data/RNG continuation.
