#!/usr/bin/env python3
"""Check that the local model directory has its required files."""
from pathlib import Path
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent / "model")
    args = p.parse_args()
    folder = args.directory
    from transformers import AutoConfig, AutoTokenizer
    if not (folder / "config.json").is_file():
        raise SystemExit("Missing config.json: first run create_model.py --size YOUR_PRESET --create")
    cfg = AutoConfig.from_pretrained(str(folder), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(folder), local_files_only=True)
    shards = list(folder.glob("*.safetensors"))
    if not shards:
        raise SystemExit("SafeTensors weights are missing: model creation is incomplete.")
    print("Architecture:", cfg.model_type)
    print("Layers:", cfg.num_hidden_layers, "| dimension:", cfg.hidden_size)
    print("Tokenizer:", len(tokenizer), "tokens")
    print("Weights: ", len(shards), "files,", round(sum(x.stat().st_size for x in shards)/1e9, 2), "GB")
    print("OK: model directory has the required files. This does not validate Unsloth or model quality.")


if __name__ == "__main__":
    main()
