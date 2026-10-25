#!/usr/bin/env python3
"""Run full-weight pretraining or supervised conversation training with checkpoints."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from model_architecture import llama_parameter_count

from text_training_common import (ShardTextStream, load_full_checkpoint,
                                      resolve_checkpoint, save_full_checkpoint)

ROOT = Path(__file__).resolve().parent


def parse_args(argv=None, *, backend='pytorch', mode='pretrain'):
    p = argparse.ArgumentParser(description='Full-weight training for champ-llm, with resumable checkpoints.')
    p.add_argument('--engine', choices=['pytorch', 'unsloth'], default=backend)
    p.add_argument('--mode', choices=['pretrain', 'conversations'], default=mode)
    p.add_argument('--model', type=Path, default=ROOT / 'model', help='Initial local blank model')
    p.add_argument('--data', type=Path, default=ROOT / 'outputs' / 'shards_manifest.json')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--steps', type=int, default=1, help='TOTAL steps, including steps completed before resuming')
    p.add_argument('--context', type=int, default=256)
    p.add_argument('--gradient-accumulation', type=int, default=1)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--save-every', type=int, default=0, help='Full checkpoint every N steps (0 = final step only)')
    p.add_argument('--resume', action='store_true', help='Resume weights, optimizer, RNG and dataset cursor')
    p.add_argument('--allow-cpu', action='store_true', help='Allow CPU training for a model with at most 100 million parameters')
    args = p.parse_args(argv)
    if args.output is None:
        args.output = ROOT / 'training_runs' / (args.engine + '_' + args.mode)
    if args.steps < 1 or args.context < 16 or args.gradient_accumulation < 1 or args.learning_rate <= 0 or args.save_every < 0:
        p.error('--steps>=1 --context>=16 --gradient-accumulation>=1 --learning-rate>0 --save-every>=0')
    if args.model.resolve() == args.output.resolve() or args.output.resolve().is_relative_to(args.model.resolve()):
        p.error('Output directory must not overwrite or be inside the initial model directory')
    if args.resume:
        if not (args.output/'latest.json').is_file():
            p.error('Cannot resume: latest.json is missing')
    elif (args.output/'latest.json').exists():
        p.error('Training already exists in this directory: use --resume or a different --output')
    return args


def run_training(args):
    import torch
    if args.engine == 'unsloth':
        
        from unsloth import FastLanguageModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    random.seed(3407)
    torch.manual_seed(3407)
    use_cuda = torch.cuda.is_available()
    if not use_cuda and not args.allow_cpu:
        raise RuntimeError('CUDA GPU required. For a small model on CPU, explicitly use --allow-cpu')
    if args.engine == 'unsloth' and not use_cuda:
        raise RuntimeError('Unsloth requires a GPU in this script')
    device = torch.device('cuda' if use_cuda else 'cpu')
    source = resolve_checkpoint(args.output) if args.resume else None
    model_dir = (source/'model') if source else args.model
    if not (model_dir/'config.json').exists():
        raise FileNotFoundError('Missing config.json: first create the blank model.')
    conf = AutoConfig.from_pretrained(str(model_dir), local_files_only=True)
    if args.context + 1 > conf.max_position_embeddings:
        raise ValueError('Requested context exceeds max_position_embeddings')
    param_estimate = (llama_parameter_count(conf.to_dict()) if conf.model_type == 'llama' else
                      getattr(conf, 'num_hidden_layers', 1) * getattr(conf, 'hidden_size', 1)**2 * 10)
    if not use_cuda and param_estimate > 100_000_000:
        raise RuntimeError('CPU training is limited to 100 million parameters; choose --size 20m or a small custom configuration')
    if use_cuda and param_estimate > 1_000_000_000 and torch.cuda.get_device_properties(0).total_memory < 64 * 1024**3:
        raise RuntimeError('The current full-model training guard requires at least 64 GiB VRAM above one billion estimated parameters; larger models may need more')
    if use_cuda and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This GPU training configuration requires BF16 support')
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)

    if args.engine == 'unsloth':
        
        
        
        from bitsandbytes.optim import AdamW8bit
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(model_dir), max_seq_length=args.context + 1,
            load_in_4bit=False, full_finetuning=True, trust_remote_code=False,
        )
        optimizer_cls = AdamW8bit
        optimizer_kwargs = {'lr': args.learning_rate, 'betas': (0.9, 0.95), 'weight_decay': 0.1}
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), local_files_only=True, dtype=torch.float32,
            attn_implementation='sdpa',
        ).to(device)
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {'lr': args.learning_rate, 'betas': (0.9, 0.95),
                            'weight_decay': 0.1, 'fused': use_cuda}

    model.train()
    model.config.use_cache = False
    if hasattr(model, 'gradient_checkpointing_enable'):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        except TypeError:
            model.gradient_checkpointing_enable()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if total != trainable:
        raise RuntimeError(f'All weights must be trainable. Total={total}; actifs={trainable}')
    print(f'{args.engine}: {total:,} fully trainable parameters on {device}', flush=True)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    data = ShardTextStream(args.data, tokenizer, mode=args.mode, context=args.context)
    arch = {name: getattr(conf, name, None) for name in (
        'model_type','vocab_size','hidden_size','intermediate_size','num_hidden_layers',
        'num_attention_heads','num_key_value_heads','max_position_embeddings','tie_word_embeddings')}
    options = {'model_architecture': arch, 'context': args.context,
               'gradient_accumulation': args.gradient_accumulation, 'learning_rate': args.learning_rate,
               'engine': args.engine, 'optimizer': optimizer_cls.__name__}
    start_step = 0
    if source:
        start_step = load_full_checkpoint(source, optimizer, data, engine=args.engine, options=options)
        if start_step >= args.steps:
            raise ValueError(f'Already at {start_step} steps; --steps must be higher')
        print(f'Resumed after step {start_step}; cursor={data.cursor["shard_index"]}:{data.cursor["offset"]}', flush=True)

    completed = start_step
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.gradient_accumulation):
            token_ids, labels = next(data)
            ids = torch.tensor([token_ids], dtype=torch.long, device=device)
            targets = torch.tensor([labels], dtype=torch.long, device=device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_cuda):
                loss = model(input_ids=ids, labels=targets, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss: check training data and learning rate')
            (loss / args.gradient_accumulation).backward()
            loss_sum += loss.detach().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        completed = step
        print(f'Step {step}/{args.steps} | loss={loss_sum/args.gradient_accumulation:.4f}', flush=True)
        if args.save_every and step % args.save_every == 0:
            save_full_checkpoint(args.output, model, tokenizer, optimizer, data,
                                 step=step, engine=args.engine, options=options)
            print('Full checkpoint saved', step, flush=True)
    if not args.save_every or completed % args.save_every:
        save_full_checkpoint(args.output, model, tokenizer, optimizer, data,
                             step=completed, engine=args.engine, options=options)
    print(f'Done. Resume: python {"train_unsloth.py" if args.engine=="unsloth" else "train_pytorch.py"} '
          f'--mode {args.mode} --steps {args.steps+1} --resume --output "{args.output}"')


def main(argv=None, *, backend='pytorch', mode='pretrain'):
    args = parse_args(argv, backend=backend, mode=mode)
    run_training(args)


if __name__ == '__main__':
    main()
