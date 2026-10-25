#!/usr/bin/env python3
"""Track vision shard positions and save/restores resumable training state."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from vision_model import VisionEncoder
from shards_io import resolve_shards
from vision_integrity import verify_vision_record


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VisionShardStream:
    """VisionShardStream provides reusable operations for this project."""
    def __init__(self, manifest: Path, project_root: Path, *, seed: int = 3407,
                 cursor: dict | None = None):
        self.manifest = Path(manifest)
        self.project_root = Path(project_root)
        self.allowed = (self.project_root / 'training_data').resolve()
        self.paths = list(resolve_shards(self.manifest, 'vision_train'))
        if not self.paths:
            raise ValueError('No training shards for vision')
        self.seed = seed
        self.cursor = {'epoch': 0, 'shard_index': 0, 'offset': 0, 'count': 0}
        if cursor:
            self.cursor.update(cursor)
            if not (0 <= self.cursor['shard_index'] < len(self.paths)):
                raise ValueError('Cursor is incompatible with the shards')

    def _order(self):
        indexes = list(range(len(self.paths)))
        random.Random(self.seed + self.cursor['epoch']).shuffle(indexes)
        return indexes

    def __next__(self):
        
        inspected = 0
        while inspected <= len(self.paths):
            idx = self._order()[self.cursor['shard_index']]
            path = self.paths[idx]
            with path.open('rb') as file:
                file.seek(self.cursor['offset'])
                line = file.readline()
                new_offset = file.tell()
            if not line:
                inspected += 1
                self.cursor['shard_index'] += 1
                self.cursor['offset'] = 0
                if self.cursor['shard_index'] >= len(self.paths):
                    self.cursor['shard_index'] = 0
                    self.cursor['epoch'] += 1
                continue
            self.cursor['offset'] = new_offset
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = verify_vision_record(record, self.project_root)
            self.cursor['count'] += 1
            return image_path, record
        raise ValueError('Vision shards contain no examples')


def checkpoint_path(output: Path, *, resume: bool, path: Path | None = None) -> Path:
    if path is not None:
        resolved = Path(path).resolve()
    elif resume:
        latest = output / 'latest.json'
        if not latest.is_file():
            raise ValueError('No checkpoint available to resume : latest.json is missing')
        name = json.loads(latest.read_text(encoding='utf-8'))['checkpoint']
        if '/' in name or '\\' in name or name.startswith('.'):
            raise ValueError('Invalid checkpoint name in latest.json')
        resolved = (output / 'checkpoints' / name).resolve()
    else:
        raise ValueError('Specify --resume or an explicit path')
    if not resolved.joinpath('state.pt').is_file():
        raise ValueError('Incomplete checkpoint : ' + str(resolved))
    return resolved


def save_checkpoint(output: Path, vision: VisionEncoder, optimizer,
                    *, step: int, cursor: dict, manifest_hash: str, mode: str,
                    model, tokenizer=None, args_config: dict | None = None):
    """Handle resumable training state: save checkpoint."""
    output = Path(output)
    ckpts = output / 'checkpoints'
    ckpts.mkdir(parents=True, exist_ok=True)
    target = ckpts / f'step-{step:08d}'
    if target.exists():
        raise FileExistsError(f'Checkpoint already exists : {target}')
    temp = Path(tempfile.mkdtemp(prefix='.checkpoint_', dir=ckpts))
    try:
        weights = {k: v.detach().float().cpu().contiguous() for k, v in vision.state_dict().items()}
        save_file(weights, str(temp/'vision.safetensors'))
        (temp/'vision_config.json').write_text(json.dumps(vision.vision_config, indent=2), encoding='utf-8')
        meta = {
            'format': 2, 'step': step, 'cursor': cursor, 'mode': mode,
            'manifest_sha256': manifest_hash, 'parameters': args_config or {},
        }
        (temp/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
        torch.save({
            'optimizer': optimizer.state_dict(),
            'rng_cpu': torch.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }, temp/'state.pt')
        if mode == 'joint':
            model.save_pretrained(str(temp/'language'), safe_serialization=True, max_shard_size='2GB')
            if tokenizer is None:
                raise ValueError('A tokenizer is required to save the joint language model')
            tokenizer.save_pretrained(str(temp/'language'))
        os.replace(temp, target)
        
        fd, tmp_name = tempfile.mkstemp(dir=output, prefix='.latest_', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'checkpoint': target.name, 'step': step}, f)
        os.replace(tmp_name, output/'latest.json')
    except Exception:
        if temp.exists():
            shutil.rmtree(temp)
        raise
    return target


def load_checkpoint(folder: Path, vision, optimizer, *, manifest_hash: str, mode: str,
                    args_config: dict | None = None, device: torch.device | str = 'cpu'):
    meta = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if meta.get('manifest_sha256') != manifest_hash:
        raise ValueError('Dataset manifest changed: resume refused to prevent a stale cursor.')
    if meta.get('mode') != mode:
        raise ValueError('Vision mode does not match checkpoint mode')
    if meta.get('parameters') != (args_config or {}):
        raise ValueError('Model or training parameters changed: resume refused.')
    if meta.get('format') != 2 or meta.get('step', 0) < 1:
        raise ValueError('Invalid checkpoint format')
    saved_conf = json.loads((folder/'vision_config.json').read_text(encoding='utf-8'))
    if saved_conf != vision.vision_config:
        raise ValueError('Vision architecture differs from the checkpoint')
    vision.load_state_dict(load_file(str(folder/'vision.safetensors')))
    data = torch.load(folder/'state.pt', map_location=device, weights_only=True)
    optimizer.load_state_dict(data['optimizer'])
    torch.set_rng_state(data['rng_cpu'].cpu())
    if torch.cuda.is_available() and data['rng_cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in data['rng_cuda']])
    return meta
