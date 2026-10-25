#!/usr/bin/env python3
"""Plan or create a chosen randomly initialized Llama model; no pretrained weights."""
import argparse
import json
from pathlib import Path

from model_architecture import Architecture, PRESETS

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / 'model'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--plan', action='store_true', help='Preview the architecture without creating files or allocating weights')
    actions.add_argument('--create', action='store_true', help='Create and save the chosen random weights using a finalized tokenizer')
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--size', choices=PRESETS, help='Approximate model size; no default is selected')
    choice.add_argument('--config', type=Path, help='JSON file containing custom architecture dimensions')
    parser.add_argument('--directory', type=Path, default=DEFAULT_OUT, help='Output model directory')
    parser.add_argument('--tokenizer', type=Path, help='Existing local tokenizer directory; otherwise use the output directory or model/')
    parser.add_argument('--vocab-size', type=int, help='Provisional vocabulary size, for --plan only (default without a tokenizer: 32768)')
    args = parser.parse_args(argv)
    if args.vocab_size is not None and args.tokenizer is not None:
        parser.error('Choose either --vocab-size for a provisional plan or --tokenizer for an actual vocabulary')
    if args.vocab_size is not None and (args.vocab_size < 1 or args.create):
        parser.error('--vocab-size must be positive and is only allowed with --plan')
    try:
        args.architecture = (Architecture.from_dict(json.loads(args.config.read_text(encoding='utf-8-sig')))
                             if args.config else PRESETS[args.size])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv=None):
    args = parse_args(argv)
    folder = args.directory.expanduser().resolve()
    if args.create:
        if folder.exists() and not folder.is_dir():
            raise SystemExit('The model output must be a directory.')
        if (folder / 'config.json').exists() or any(folder.glob('*.safetensors')) or any(folder.glob('*.bin')):
            raise SystemExit('A model already exists in this directory; nothing was overwritten.')

    tokenizer_source = args.tokenizer.expanduser().resolve() if args.tokenizer else (
        folder if (folder / 'tokenizer.json').is_file() else DEFAULT_OUT.resolve())
    tokenizer = None
    if args.vocab_size is None and (tokenizer_source / 'tokenizer.json').is_file():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source), local_files_only=True, trust_remote_code=False)
        vocab_size = len(tokenizer)
        vocab_note = 'finalized local tokenizer'
    else:
        if args.create or args.tokenizer is not None:
            raise SystemExit('Tokenizer missing. Run prepare_tokenizer.py or pass --tokenizer with a local tokenizer directory.')
        vocab_size = args.vocab_size if args.vocab_size is not None else 32768
        vocab_note = 'provisional; run the plan again with your finalized tokenizer'

    architecture = args.architecture
    count = architecture.parameter_count(vocab_size)
    profile = args.size or 'custom'
    print(f'Profile: {profile} | Vocabulary: {vocab_size:,} ({vocab_note})')
    print(f'Layers: {architecture.num_hidden_layers} | Hidden size: {architecture.hidden_size} | MLP size: {architecture.intermediate_size}')
    print(f'Attention heads: {architecture.num_attention_heads} | KV heads: {architecture.num_key_value_heads} | Context: {architecture.max_position_embeddings}')
    print(f'Parameters: {count:,} ({count / 1e9:.3f} billion; text model only)')
    print(f'BF16 weights alone: approximately {count * 2 / 1e9:.2f} GB')
    print(f'FP32 PyTorch weights, gradients and AdamW states alone: approximately {count * 16 / 1e9:.2f} GB')
    print('Training also needs activations and runtime memory; these are not total VRAM requirements.')
    if args.plan:
        print('Plan only: no files or weights created. Finalize the tokenizer, then use --create with the same --size or --config.')
        return

    if folder != tokenizer_source and folder.exists() and any(folder.iterdir()):
        raise SystemExit('A separate model output directory must be empty; nothing was overwritten.')
    if any(getattr(tokenizer, name) is None for name in ('bos_token_id', 'eos_token_id', 'pad_token_id')):
        raise SystemExit('The tokenizer must define BOS, EOS and PAD token IDs.')
    from transformers import LlamaConfig, LlamaForCausalLM
    import torch
    config = LlamaConfig(
        **architecture.to_dict(), vocab_size=vocab_size,
        hidden_act='silu', rms_norm_eps=1e-6, rope_theta=10000,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id, tie_word_embeddings=False,
        attention_bias=False, mlp_bias=False, use_cache=True,
    )
    config._name_or_path = 'champ-llm'
    torch.manual_seed(3407)
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = LlamaForCausalLM(config)
    finally:
        torch.set_default_dtype(before)
    actual = sum(p.numel() for p in model.parameters())
    if actual != count:
        raise RuntimeError(f'Parameter count mismatch: planned {count}, constructed {actual}; model not saved.')
    model.save_pretrained(str(folder), safe_serialization=True, max_shard_size='2GB')
    tokenizer.model_max_length = architecture.max_position_embeddings
    tokenizer.save_pretrained(str(folder))
    (folder / 'model_identity.json').write_text(json.dumps({
        'name': 'champ-llm', 'origin': 'randomly initialized weights',
        'params': actual, 'architecture': 'llama-decoder', 'profile': profile,
        'dimensions': architecture.to_dict(), 'vocab_size': vocab_size,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Blank model saved to {folder}: {actual:,} parameters. No training has occurred.')


if __name__ == '__main__':
    main()
