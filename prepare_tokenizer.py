#!/usr/bin/env python3
"""Train a local tokenizer from previously prepared text shards."""
from __future__ import annotations
import argparse
from pathlib import Path
from shards_io import resolve_shards

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'model'
MANIFEST = ROOT / 'outputs' / 'shards_manifest.json'
SPECIAL = [
    '<pad>', '<s>', '</s>', '<unk>',
    '<|system|>', '<|user|>', '<|assistant|>', '<|tool|>', '<|end|>',
    '<|file_start|>', '<|file_end|>',
]


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build a local tokenizer from previously prepared shards.')
    parser.add_argument('--vocab-size', type=int, default=32768)
    parser.add_argument('--data', type=Path, default=MANIFEST,
                        help='JSON manifest, tokenizer shard directory or a legacy corpus text file')
    args = parser.parse_args(argv)
    if args.vocab_size < 1000:
        parser.error('--vocab-size must be at least 1000')
    if (OUT / 'config.json').exists() or (OUT / 'model.safetensors').exists() or list(OUT.glob('model-*.safetensors')):
        raise SystemExit('A model already exists: keep the tokenizer tied to its token IDs.')
    paths = resolve_shards(args.data, 'tokenizer')
    total = sum(p.stat().st_size for p in paths)
    if total < 1_000_000:
        print('WARNING: corpus under 1 MiB is suitable for testing, not for a useful language model.')
    print(f'Training tokenizer on {len(paths)} shards and {total/1024**2:.1f} MiB.', flush=True)

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast
    tok = Tokenizer(models.BPE(unk_token='<unk>'))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=2,
        special_tokens=SPECIAL, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    def documents():
        
        for path in paths:
            with path.open('r', encoding='utf-8') as stream:
                while True:
                    part = stream.read(65536)
                    if not part:
                        break
                    yield part

    tok.train_from_iterator(documents(), trainer=trainer)
    hf = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token='<s>', eos_token='</s>', pad_token='<pad>', unk_token='<unk>',
        additional_special_tokens=SPECIAL[4:],
        model_max_length=4096, clean_up_tokenization_spaces=False,
    )
    hf.chat_template = (
        "{% for message in messages %}"
        "<|{{ message['role'] }}|>\n{{ message['content'] }}\n<|end|>\n"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(str(OUT))
    phrase = 'Bonjour ! function Bonjour(){return <h1>Salut !</h1>}'
    ids = hf(phrase, add_special_tokens=False)['input_ids']
    assert hf.decode(ids, skip_special_tokens=False) == phrase
    print(f'Tokenizer created : {len(hf):,} tokens in {OUT}')
    print('Do not recreate this tokenizer after model weights have been initialized.')


if __name__ == '__main__':
    main()
