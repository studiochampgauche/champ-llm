#!/usr/bin/env python3
"""Tools for the champ-llm local training pipeline."""
import json
from train_text import main


def chunks_from_jsonl(paths, tokenizer, context):
    """Implement chunks from jsonl for the local model pipeline."""
    buffer = []
    for path in paths:
        with path.open(encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get('text')
                if not isinstance(text, str) or not text.strip():
                    continue
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if tokenizer.eos_token_id is not None:
                    tokens.append(tokenizer.eos_token_id)
                buffer.extend(tokens)
                while len(buffer) >= context+1:
                    yield buffer[:context+1]
                    del buffer[:context]


if __name__ == '__main__':
    main(backend='pytorch')
