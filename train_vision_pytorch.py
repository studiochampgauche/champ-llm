#!/usr/bin/env python3
"""Train a custom vision adapter with resumable checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from vision_model import VisionEncoder, image_to_tiles, assemble_example, multimodal_loss
from vision_checkpoints import (VisionShardStream, manifest_sha256, checkpoint_path,
                           save_checkpoint, load_checkpoint)

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='Train a randomly initialized vision adapter and save full checkpoints for resuming.')
    p.add_argument('--model', type=Path, default=ROOT/'model')
    p.add_argument('--manifest', type=Path, default=ROOT/'outputs'/'vision'/'vision_manifest.json')
    p.add_argument('--output', type=Path, default=ROOT/'training_runs'/'vision_trial')
    p.add_argument('--steps', type=int, default=1, help='NUMBER OF ADDITIONAL STEPS, including when resuming')
    p.add_argument('--resume', action='store_true', help='Resume last checkpoint, including optimizer and dataset cursor')
    p.add_argument('--checkpoint', type=Path, help='Resume from this checkpoint (alternative to --resume)')
    p.add_argument('--save-every', type=int, default=10)
    p.add_argument('--mode', choices=['vision-only', 'joint'], default='vision-only')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=3407)
    p.add_argument('--image-size', type=int, choices=[32, 112, 224, 336, 448], default=224)
    p.add_argument('--max-tiles', type=int, default=9)
    p.add_argument('--pool-grid', type=int, default=4)
    p.add_argument('--vision-dim', type=int, default=256)
    p.add_argument('--vision-layers', type=int, default=2)
    p.add_argument('--max-prompt-tokens', type=int, default=384)
    p.add_argument('--max-answer-tokens', type=int, default=2048)
    p.add_argument('--allow-truncation', action='store_true',
                   help='Explicitly truncate long examples (not recommended for source code)')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if a.steps < 1 or a.save_every < 1 or a.max_tiles < 1 or a.lr <= 0 or a.pool_grid < 1:
        p.error('Require steps/save-every/max-tiles >= 1 and lr > 0')
    if a.max_prompt_tokens < 1 or a.max_answer_tokens < 2:
        p.error('Invalid token limits')
    return a


def config_training(args):
    """Implement config training for the local model pipeline."""
    config = json.loads((args.model/'config.json').read_text(encoding='utf-8'))
    token_path = args.model/'tokenizer.json'
    return {
        'model_config': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        'tokenizer_sha': hashlib.sha256(token_path.read_bytes()).hexdigest() if token_path.exists() else 'missing',
        'mode': args.mode, 'lr': args.lr, 'seed': args.seed,
        'image_size': args.image_size, 'max_tiles': args.max_tiles,
        'pool_grid': args.pool_grid, 'vision_dim': args.vision_dim,
        'vision_layers': args.vision_layers, 'max_prompt_tokens': args.max_prompt_tokens,
        'max_answer_tokens': args.max_answer_tokens, 'allow_truncation': args.allow_truncation,
    }


def train(args, *, load_language=None):
    if not args.manifest.is_file():
        raise ValueError('Missing vision_manifest.json: first run prepare_vision.py')
    hash_manifest = manifest_sha256(args.manifest)
    stream = VisionShardStream(args.manifest, ROOT, seed=args.seed)
    if args.dry_run:
        for _ in range(3):
            image, example = next(stream)
            print('Image :', image, '| question :', example['question'][:70],
                  '| answer (characters):', len(example['answer']))
        print('No language model is loaded or trained. Requested dimensions:', args.image_size,
              'pixels per tile,', args.max_tiles, 'tiles.')
        return None
    if not (args.model/'config.json').is_file():
        raise ValueError('Local model is missing: create a blank model, then pretrain it on language and code.')
    if args.mode == 'joint' and not torch.cuda.is_available():
        raise ValueError('Joint language and vision training requires a sufficiently powerful GPU')
    settings = config_training(args)
    resume = args.resume or args.checkpoint is not None
    resumed_ckpt = checkpoint_path(args.output, resume=True, path=args.checkpoint) if resume else None
    source = resumed_ckpt/'language' if resumed_ckpt and args.mode == 'joint' else args.model
    if not resume and args.output.joinpath('latest.json').exists():
        raise ValueError('Checkpoints already exist. Add --resume or use a different --output.')

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float32
    if load_language:
        model, tokenizer = load_language(source, device, dtype)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(source), local_files_only=True, torch_dtype=dtype,
            trust_remote_code=False, low_cpu_mem_usage=True,
        ).to(device)
    if hasattr(model, 'config'):
        model.config.use_cache = False
    if args.mode == 'vision-only':
        model.requires_grad_(False).eval()
    else:
        model.train()
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
    
    vision = VisionEncoder(model.config.hidden_size, image_size=args.image_size,
                           vision_dim=args.vision_dim, vision_layers=args.vision_layers,
                           max_tiles=args.max_tiles, pool_grid=args.pool_grid).to(device)
    trainables = list(vision.parameters())
    if args.mode == 'joint':
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError('Joint training requires bitsandbytes and a compatible GPU.') from e
        trainables.extend(p for p in model.parameters() if p.requires_grad)
        optimizer = bnb.optim.AdamW8bit(trainables, lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(trainables, lr=args.lr)
    start_step = 0
    if resumed_ckpt:
        restored = load_checkpoint(resumed_ckpt, vision, optimizer,
                                   manifest_hash=hash_manifest, mode=args.mode,
                                   args_config=settings, device=device)
        start_step = restored['step']
        stream = VisionShardStream(args.manifest, ROOT, seed=args.seed, cursor=restored['cursor'])
        print(f'Full resume from step {start_step}; optimizer, RNG and data cursor restored.', flush=True)

    vision.train()
    for step in range(start_step + 1, start_step + args.steps + 1):
        image_path, row = next(stream)
        ids, labels = assemble_example(tokenizer, row['question'], row['answer'],
                                       max_prompt=args.max_prompt_tokens,
                                       max_answer=args.max_answer_tokens,
                                       allow_truncation=args.allow_truncation)
        ids = ids.unsqueeze(0).to(device)
        labels = labels.unsqueeze(0).to(device)
        images = image_to_tiles(image_path, args.image_size, args.max_tiles).unsqueeze(0).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = multimodal_loss(model, vision, images, ids, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss; no checkpoint was saved for this step.')
        loss.backward()
        optimizer.step()
        print(f'Step {step}: loss={loss.item():.5f}, '
              f'image={images.shape[1]} tiles, text tokens={ids.shape[1]}', flush=True)
        if step % args.save_every == 0 or step == start_step + args.steps:
            target = save_checkpoint(args.output, vision, optimizer, step=step,
                                     cursor=dict(stream.cursor), manifest_hash=hash_manifest,
                                     mode=args.mode, model=model, tokenizer=tokenizer,
                                     args_config=settings)
            print('Full checkpoint:', target, flush=True)

    print('Completed. A large annotated dataset is required for meaningful image understanding.')
    return args.output


def main(argv=None):
    try:
        train(parse_args(argv))
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        raise SystemExit('Error: ' + str(e)) from e


if __name__ == '__main__':
    main()
