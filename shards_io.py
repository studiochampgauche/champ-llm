"""Resolve versioned shard manifests securely and without remote access."""
from __future__ import annotations
import json
from pathlib import Path

VALID_SPLITS = frozenset({
    'pretrain_train', 'pretrain_validation',
    'conversations_train', 'conversations_validation', 'tokenizer',
    'vision_train', 'vision_validation',
})


def resolve_shards(source: str | Path, split: str = 'pretrain_train') -> list[Path]:
    """Handle streaming training shards: resolve shards."""
    source = Path(source).expanduser().resolve()
    if split not in VALID_SPLITS:
        raise ValueError(f'Split inconnu: {split}')
    if source.is_file() and source.name.endswith('.json'):
        data = json.loads(source.read_text(encoding='utf-8'))
        if data.get('version') not in (1, 2) or 'splits' not in data:
            raise ValueError('Incompatible manifest; rebuild data with prepare_data.py')
        if (split.startswith('vision_')) != (data.get('version') == 1):
            raise ValueError('Split is incompatible with the manifest type')
        root = source.parent
        paths = []
        for row in data['splits'][split]['files']:
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError(f'Unsafe path in manifest : {relative}')
            resolved = (root / relative).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError('Path escapes the manifest directory')
            paths.append(resolved)
    elif source.is_dir():
        folder = source / split if (source / split).is_dir() else source
        ext = '.txt' if split == 'tokenizer' else '.jsonl'
        paths = sorted(folder.glob(f'shard-*{ext}'))
    elif source.is_file() and source.suffix in {'.jsonl', '.txt'}:
        paths = [source]
    else:
        raise FileNotFoundError(f'Data source not found: {source}')
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f'Declared shard not found: {p}')
    if not paths:
        raise ValueError(f'No shards found for split {split} in {source}')
    return paths
