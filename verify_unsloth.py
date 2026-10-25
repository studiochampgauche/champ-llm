#!/usr/bin/env python3
"""Probe whether Unsloth can load the randomly initialized local model."""
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "model")
    args = p.parse_args()
    folder = args.directory.resolve()
    if not (folder / "config.json").exists():
        raise SystemExit("Model not found: first run create_model.py --size YOUR_PRESET --create")
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This Unsloth loading check requires a CUDA GPU; use RunPod or similar hardware.")
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(folder), max_seq_length=256,
            load_in_4bit=False, full_finetuning=True,
            trust_remote_code=False,
        )
        print("SUCCESS: Unsloth loaded this local model.")
        print("Parameters:", sum(p.numel() for p in model.parameters()))
        print("This is NOT a training test. Dataset setup and a training step are still required.")
    except Exception as exc:
        print("Unsloth could not open this checkpoint as a trainable model:")
        print(type(exc).__name__ + ": " + str(exc)[:1500])
        print("This does not imply unusable weights: standard Transformers models may be trained with PyTorch.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
