# GPU training on RunPod

Review your model's parameter count and memory estimates in
[Model configuration](docs/MODEL_SIZES.md) before choosing a GPU.
Begin with a short run to verify loading, training, saving and resumption.

## Prepare the environment

Choose a machine with a compatible NVIDIA GPU and enough memory and storage.
Use a supported PyTorch/CUDA environment with Python 3.13, and persistent storage
for data and checkpoints.

Copy the project, prepared `outputs/` shards, and `model/` weights and tokenizer
to the machine. From the project directory:

```sh
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt
```

## Check training with PyTorch

```sh
python train_pytorch.py --mode pretrain --steps 1 --context 128
```

Check that the loss is finite and that a complete checkpoint is saved. Memory
requirements include activations and optimizer state as well as weights.
Resolve out-of-memory errors before extending the run.

## Resume the run

```sh
python train_pytorch.py --mode pretrain --steps 2 --context 128 --resume
```

For text training, `--steps` is the total target: this command continues to step 2.
Keep the dataset manifest, referenced shards and compatible training settings
unchanged. Back up complete checkpoints to preserve the optimizer state and
dataset position.

Once these checks pass, choose a training schedule and evaluate on held-out data.
Full training at billion-parameter scale remains unvalidated in the project's
recorded tests.

## Optional Unsloth backend

Install Unsloth in an environment that satisfies its dependency requirements, then
check loading and a short training run:

```sh
python -m pip install unsloth datasets trl bitsandbytes
python verify_unsloth.py --directory model
python train_unsloth.py --mode pretrain --steps 1 --context 128
python train_unsloth.py --mode pretrain --steps 2 --context 128 --resume
```

Full-weight Unsloth training of the project's custom randomly initialized models
remains experimental. A successful load alone does not validate training or
checkpoint resumption.

## Add further training stages

Use [conversation training](CONVERSATIONS_CHECKPOINTS_AGENT.md) after pretraining.
For annotated screenshots, follow [Vision and browser](VISION_AND_BROWSER.md).
Vision training counts `--steps` as additional steps, including when resuming.
