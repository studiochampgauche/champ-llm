#!/usr/bin/env python3
"""Consistent conversation tokenization and resumable text training checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

from shards_io import resolve_shards

ROLES = frozenset({'system', 'user', 'assistant', 'tool'})


def _message_parts(tokenizer, role: str, content: str) -> tuple[list[int], list[int], list[int]]:
    """Handle role-aligned tokenization for  message parts."""
    return (tokenizer.encode(f'<|{role}|>\n', add_special_tokens=False),
            tokenizer.encode(content, add_special_tokens=False),
            tokenizer.encode('\n<|end|>\n', add_special_tokens=False))


def assistant_prompt_tokens(tokenizer, question: str) -> list[int]:
    """Handle role-aligned tokenization for assistant prompt tokens."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Prompt must contain nonempty text')
    ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    header, body, ending = _message_parts(tokenizer, 'user', question)
    ids.extend(header + body + ending)
    ids.extend(tokenizer.encode('<|assistant|>\n', add_special_tokens=False))
    return ids


def conversation_tokens(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Handle role-aligned tokenization for conversation tokens."""
    if not isinstance(messages, list) or not messages:
        raise ValueError('messages must be a nonempty list')
    ids: list[int] = []
    labels: list[int] = []
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        labels.append(-100)
    has_supervised = False
    for msg in messages:
        if not isinstance(msg, dict) or msg.get('role') not in ROLES or not isinstance(msg.get('content'), str):
            raise ValueError('Invalid message: role and content are required')
        role = msg['role']
        learn = role == 'assistant' and msg.get('learn', True) is not False
        if 'learn' in msg and not isinstance(msg['learn'], bool):
            raise ValueError('learn must be a boolean')
        header, body, ending = _message_parts(tokenizer, role, msg['content'])
        ids.extend(header + body + ending)
        labels.extend([-100]*len(header))
        labels.extend(body if learn else [-100]*len(body))
        labels.extend(ending if learn else [-100]*len(ending))
        has_supervised |= learn and bool(body)
    if not has_supervised:
        raise ValueError('No assistant response marked for training')
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id if messages[-1]['role'] == 'assistant' and
                      messages[-1].get('learn', True) is not False else -100)
    assert len(ids) == len(labels)
    return ids, labels


def fingerprint_source(manifest: Path, shards: list[Path]) -> str:
    """Detect manifest/path/size/mtime changes without rereading huge corpora.

    This is not a cryptographic hash of shard contents. Do not modify a dataset
    while a training run or its resumable checkpoints still depend on it."""
    h = hashlib.sha256()
    source = Path(manifest).resolve()
    h.update(str(source).encode())
    if source.is_file() and source.suffix == '.json':
        h.update(source.read_bytes())
    for path in shards:
        stat = path.stat()
        h.update(f'{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n'.encode())
    return h.hexdigest()


class ShardTextStream:
    """ShardTextStream provides reusable operations for this project."""
    def __init__(self, manifest: Path, tokenizer, *, mode: str, context: int, seed: int = 3407,
                 cursor: dict | None = None):
        if mode not in ('pretrain', 'conversations') or context < 2:
            raise ValueError('Invalid mode/context')
        self.mode = mode
        self.context = context
        self.tokenizer = tokenizer
        self.manifest = Path(manifest).resolve()
        self.paths = resolve_shards(self.manifest, 'pretrain_train' if mode == 'pretrain' else 'conversations_train')
        self.fingerprint = fingerprint_source(self.manifest, self.paths)
        self.seed = seed
        self.cursor = {'epoch': 0, 'shard_index': 0, 'offset': 0,
                       'buffer_ids': [], 'buffer_labels': [], 'examples': 0}
        if cursor is not None:
            self.cursor.update(cursor)
            if not (0 <= self.cursor['shard_index'] < len(self.paths)) or self.cursor['offset'] < 0:
                raise ValueError('Incompatible cursor')

    def _path_order(self) -> list[Path]:
        paths = list(self.paths)
        random.Random(self.seed + self.cursor['epoch']).shuffle(paths)
        return paths

    def _read_record(self) -> dict:
        skipped_files = 0
        while skipped_files <= len(self.paths):
            p = self._path_order()[self.cursor['shard_index']]
            with p.open('rb') as f:
                f.seek(self.cursor['offset'])
                line = f.readline()
                new_offset = f.tell()
            if not line:
                skipped_files += 1
                self.cursor['shard_index'] += 1
                self.cursor['offset'] = 0
                if self.cursor['shard_index'] == len(self.paths):
                    self.cursor['shard_index'] = 0
                    self.cursor['epoch'] += 1
                continue
            self.cursor['offset'] = new_offset
            if not line.strip():
                continue
            return json.loads(line.decode('utf-8'))
        raise ValueError('No nonempty records in the shards')

    def _feed(self):
        record = self._read_record()
        if self.mode == 'pretrain':
            content = record.get('text')
            if not isinstance(content, str) or not content.strip():
                return
            tokens = self.tokenizer.encode(content, add_special_tokens=False)
            if self.tokenizer.eos_token_id is not None:
                tokens.append(self.tokenizer.eos_token_id)
            self.cursor['buffer_ids'].extend(tokens)
            self.cursor['buffer_labels'].extend(tokens)
        else:
            tokens, labels = conversation_tokens(self.tokenizer, record.get('messages'))
            self.cursor['buffer_ids'] = tokens
            self.cursor['buffer_labels'] = labels
        self.cursor['examples'] += 1

    def __next__(self):
        
        for _ in range(10000):
            ids = self.cursor['buffer_ids']
            labels = self.cursor['buffer_labels']
            if self.mode == 'pretrain':
                if len(ids) < self.context + 1:
                    self._feed()
                    continue
                length = self.context + 1
                result = (ids[:length], labels[:length])
                del ids[:self.context]
                del labels[:self.context]
                return result
            
            if len(ids) < 2:
                self._feed()
                continue
            length = min(len(ids), self.context + 1)
            result = (ids[:length], labels[:length])
            if len(ids) <= length:
                ids.clear()
                labels.clear()
            else:
                del ids[:self.context]
                del labels[:self.context]
            if any(x != -100 for x in result[1][1:]):
                return result
        raise ValueError('Too many sequences without supervised tokens; check the dataset')

    def state_dict(self):
        return json.loads(json.dumps(self.cursor))


def resolve_checkpoint(output: Path) -> Path:
    latest = output / 'latest.json'
    if not latest.is_file():
        raise ValueError(f'No checkpoint available to resume in {output}')
    obj = json.loads(latest.read_text(encoding='utf-8'))
    name = obj.get('checkpoint', '')
    if not isinstance(name, str) or not name.startswith('step-') or '/' in name or '\\' in name:
        raise ValueError('Invalid latest.json')
    folder = (output / 'checkpoints' / name).resolve()
    if not folder.is_relative_to(output.resolve()) or not (folder / 'state.pt').is_file():
        raise ValueError('Missing or incomplete checkpoint')
    return folder


def save_full_checkpoint(output: Path, model, tokenizer, optimizer, data: ShardTextStream, *,
                         step: int, engine: str, options: dict):
    """Handle resumable training state: save full checkpoint."""
    import torch
    output = Path(output).resolve()
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(parents=True, exist_ok=True)
    target = checkpoints / f'step-{step:08d}'
    if target.exists():
        raise FileExistsError(f'Checkpoint already exists: {target}')
    temp = Path(tempfile.mkdtemp(prefix='.incomplete-', dir=checkpoints))
    try:
        model_dir = temp / 'model'
        model.save_pretrained(str(model_dir), safe_serialization=True, max_shard_size='2GB')
        tokenizer.save_pretrained(str(model_dir))
        torch.save({'optimizer': optimizer.state_dict(),
                    'rng_cpu': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    'rng_python': random.getstate(),
                    'cursor': data.state_dict()}, temp / 'state.pt')
        meta = {'format': 1, 'step': step, 'engine': engine, 'mode': data.mode,
                'fingerprint': data.fingerprint, 'options': options}
        (temp / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, target)
        fd, name = tempfile.mkstemp(prefix='.latest-', suffix='.json', dir=output)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'checkpoint': target.name, 'step': step}, f)
        os.replace(name, output / 'latest.json')
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return target


def load_full_checkpoint(folder: Path, optimizer, data: ShardTextStream, *, engine: str, options: dict):
    """Handle resumable training state: load full checkpoint."""
    import torch
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    if (meta.get('format') != 1 or meta.get('fingerprint') != data.fingerprint or
        meta.get('engine') != engine or meta.get('mode') != data.mode or meta.get('options') != options):
        raise ValueError('Checkpoint is incompatible: model, parameters or data changed')
    state = torch.load(folder / 'state.pt', map_location='cpu', weights_only=True)
    optimizer.load_state_dict(state['optimizer'])
    data.cursor = state['cursor']
    torch.set_rng_state(state['rng_cpu'].cpu())
    if torch.cuda.is_available() and state['rng_cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['rng_cuda']])
    random.setstate(state['rng_python'])
    return meta['step']
