#!/usr/bin/env python3
"""Ask a trained local multimodal model a question about an image."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from vision_model import VisionEncoder, image_to_tiles, generate_answer
from vision_checkpoints import checkpoint_path


def analyze(vision_dir: Path, image_path: Path, question: str, model, tokenizer,
             *, max_new_tokens: int = 120) -> str:
    if not image_path.is_file():
        raise ValueError('Image not found: ' + str(image_path))
    conf = json.loads((vision_dir/'vision_config.json').read_text(encoding='utf-8'))
    if conf.get('version') != 2:
        raise ValueError('Legacy vision checkpoint is incompatible with tiled vision v2.')
    if conf['llm_hidden'] != model.config.hidden_size:
        raise ValueError('Language and vision model dimensions do not match.')
    bridge = VisionEncoder(conf['llm_hidden'], image_size=conf['image_size'],
                           patch_size=conf['patch_size'], vision_dim=conf['vision_dim'],
                           vision_layers=conf['vision_layers'], max_tiles=conf['max_tiles'],
                           pool_grid=conf['pool_grid'])
    bridge.load_state_dict(load_file(str(vision_dir/'vision.safetensors')))
    device = next(model.parameters()).device
    bridge = bridge.to(device=device).eval()
    model.eval()
    image = image_to_tiles(image_path, conf['image_size'], conf['max_tiles'])
    image = image.unsqueeze(0).to(device=device)
    return generate_answer(model, bridge, tokenizer, image, question,
                           max_new_tokens=max_new_tokens)


def main():
    p = argparse.ArgumentParser(description='Analyze an image with a trained multimodal adapter.')
    p.add_argument('--model', type=Path, required=True, help='Trained language model')
    p.add_argument('--vision', type=Path, required=True,
                   help='Run directory with latest.json, or a checkpoint containing vision.safetensors')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--question', required=True)
    p.add_argument('--max-new-tokens', type=int, default=120)
    args = p.parse_args()
    vision_dir = args.vision
    if (vision_dir/'latest.json').exists():
        vision_dir = checkpoint_path(vision_dir, resume=True)
    if not (vision_dir/'vision.safetensors').is_file():
        p.error('Vision weights are missing; first run train_vision_pytorch.py')
    if not (args.model/'config.json').is_file():
        p.error('Language model not found')
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                                  torch_dtype=dtype, trust_remote_code=False,
                                                  low_cpu_mem_usage=True).to(device)
    print(analyze(vision_dir, args.image, args.question, model, tokenizer,
                  max_new_tokens=args.max_new_tokens))
    print('\nPrototype: generating an answer does not prove visual understanding.')


if __name__ == '__main__':
    main()
