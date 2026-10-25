#!/usr/bin/env python3
"""Experimental tiled vision encoder and multimodal generation for the local LLM."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from vision_integrity import verify_vision_record


class VisionEncoder(nn.Module):
    """VisionEncoder provides reusable operations for this project."""

    def __init__(self, llm_hidden: int, *, image_size: int = 224,
                 patch_size: int = 16, vision_dim: int = 256,
                 vision_layers: int = 2, max_tiles: int = 9, pool_grid: int | None = None):
        super().__init__()
        if image_size < 32 or image_size % patch_size:
            raise ValueError('image_size must be >=32 and divisible by patch_size')
        if pool_grid is None:
            pool_grid = min(4, image_size // patch_size)
        if max_tiles < 1 or pool_grid < 1 or pool_grid > image_size // patch_size:
            raise ValueError('Invalid max_tiles or pool_grid')
        if vision_dim % 4:
            raise ValueError('vision_dim must be divisible by 4')
        n_patches = (image_size // patch_size) ** 2
        self.image_size = image_size
        self.patch_size = patch_size
        self.max_tiles = max_tiles
        self.pool_grid = pool_grid
        self.projection_patch = nn.Conv2d(3, vision_dim, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.randn(1, n_patches, vision_dim) * 0.02)
        self.position_tuile = nn.Parameter(torch.randn(1, max_tiles, 1, vision_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(vision_dim, nhead=4, dim_feedforward=vision_dim * 4,
                                           batch_first=True, norm_first=True, dropout=0.0)
        self.transformer = nn.TransformerEncoder(layer, num_layers=vision_layers,
                                                 enable_nested_tensor=False)
        self.norm = nn.LayerNorm(vision_dim)
        self.projecteur = nn.Linear(vision_dim, llm_hidden)
        self.vision_config = {
            'version': 2, 'image_size': image_size, 'patch_size': patch_size,
            'vision_dim': vision_dim, 'vision_layers': vision_layers, 'llm_hidden': llm_hidden,
            'max_tiles': max_tiles, 'pool_grid': pool_grid,
            'tokens_per_tile': pool_grid * pool_grid,
        }

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 4:  
            images = images.unsqueeze(1)
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError('Expected [B, T, 3, H, W]')
        b, n, c, h, w = images.shape
        if n > self.max_tiles or (h, w) != (self.image_size, self.image_size):
            raise ValueError('Tiles are incompatible with the vision configuration')
        patches = self.projection_patch(images.reshape(b * n, c, h, w))
        positions = patches.flatten(2).transpose(1, 2) + self.position
        features = self.norm(self.transformer(positions))
        grid_hw = self.image_size // self.patch_size
        features = features.transpose(1, 2).reshape(b*n, -1, grid_hw, grid_hw)
        features = F.adaptive_avg_pool2d(features, (self.pool_grid, self.pool_grid))
        features = features.flatten(2).transpose(1, 2)
        features = features.reshape(b, n, self.pool_grid ** 2, -1)
        features = features + self.position_tuile[:, :n]
        return self.projecteur(features.reshape(b, n * self.pool_grid ** 2, -1))


def _to_tensor(image: Image.Image, size: int) -> torch.Tensor:
    square = Image.new('RGB', (size, size), (255, 255, 255))
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    square.paste(image, ((size-image.width)//2, (size-image.height)//2))
    pixels = np.asarray(square, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).float().div_(255.0)


def image_to_tensor(path: Path, size: int) -> torch.Tensor:
    """Handle visual data or model processing: image to tensor."""
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert('RGB')
        return _to_tensor(image, size)


def image_to_tiles(path: Path, size: int = 224, max_tiles: int = 9) -> torch.Tensor:
    """Handle visual data or model processing: image to tiles."""
    if max_tiles < 1 or size < 32:
        raise ValueError('Require max_tiles >= 1 and size >= 32')
    with Image.open(path) as original:
        im = ImageOps.exif_transpose(original).convert('RGB')
        w, h = im.size
        tiles = [_to_tensor(im.copy(), size)]
        if max_tiles == 1:
            return torch.stack(tiles)
        
        xs = sorted(set(min(i * size, max(0, w-size)) for i in range(math.ceil(w / size))))
        ys = sorted(set(min(i * size, max(0, h-size)) for i in range(math.ceil(h / size))))
        coords = [(x,y) for y in ys for x in xs]
        n = min(max_tiles-1, len(coords))
        if n == len(coords):
            sampled = coords
        elif n == 1:
            sampled = [coords[len(coords)//2]]
        else:
            sampled = [coords[round(i*(len(coords)-1)/(n-1))] for i in range(n)]
        for x,y in sampled:
            crop = im.crop((x,y,min(x+size,w),min(y+size,h)))
            tiles.append(_to_tensor(crop, size))
        return torch.stack(tiles)


def assemble_example(tokenizer, question: str, answer: str, *, max_prompt: int = 384,
                     max_answer: int = 2048, allow_truncation: bool = False):
    """Implement assemble example for the local model pipeline."""
    prompt = f'User: {question}\nAssistant:'
    pids = tokenizer.encode(prompt, add_special_tokens=False)
    aids = tokenizer.encode(answer, add_special_tokens=False)
    if len(pids) > max_prompt or len(aids) + (tokenizer.eos_token_id is not None) > max_answer:
        if not allow_truncation:
            raise ValueError(
                f'Example too long : prompt={len(pids)} tokens, answer={len(aids)} tokens; '
                f'limits={max_prompt}/{max_answer}. Increase --max-prompt-tokens / '
                '--max-answer-tokens or split this example; nothing was truncated.')
    pids = pids[:max_prompt]
    aids = aids[:max_answer - int(tokenizer.eos_token_id is not None)]
    if tokenizer.eos_token_id is not None:
        aids.append(tokenizer.eos_token_id)
    if not aids:
        raise ValueError('Empty answer')
    return torch.tensor(pids + aids, dtype=torch.long), torch.tensor([-100]*len(pids) + aids, dtype=torch.long)


def multimodal_loss(language_model, vision: VisionEncoder, images, ids, labels):
    if images.ndim not in (4, 5) or ids.ndim != 2 or labels.shape != ids.shape:
        raise ValueError('Expected images [B,T,3,H,W] and ids/labels [B,L]')
    image_embeds = vision(images)
    txt_embeds = language_model.get_input_embeddings()(ids)
    all_embeds = torch.cat([image_embeds.to(txt_embeds.dtype), txt_embeds], dim=1)
    labels = torch.cat([torch.full((ids.shape[0], image_embeds.shape[1]), -100,
                                    device=labels.device, dtype=labels.dtype), labels], dim=1)
    available_context = getattr(getattr(language_model, 'config', None), 'max_position_embeddings', None)
    if available_context and labels.shape[1] > available_context:
        raise ValueError(f'Context {labels.shape[1]} > model limit {available_context}. '
                         'Reduce image tiles/tokens or train a model with a longer context.')
    warning = torch.ones(labels.shape, dtype=torch.long, device=labels.device)
    return language_model(inputs_embeds=all_embeds, attention_mask=warning, labels=labels).loss


def generate_answer(language_model, vision, tokenizer, images, question: str, *, max_new_tokens: int = 120):
    """Implement generate answer for the local model pipeline."""
    if max_new_tokens < 1:
        raise ValueError('max_new_tokens must be positive')
    prompt = f'User: {question}\nAssistant:'
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    device = next(language_model.parameters()).device
    ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    with torch.inference_mode():
        prefix = vision(images)
        embeddings = language_model.get_input_embeddings()(ids)
        embeds = torch.cat([prefix.to(embeddings.dtype), embeddings], dim=1)
        context_limit = getattr(getattr(language_model, 'config', None), 'max_position_embeddings', None)
        if context_limit and embeds.shape[1] + max_new_tokens > context_limit:
            raise ValueError('Insufficient context for image tiles, question and answer.')
        mask = torch.ones(embeds.shape[:2], dtype=torch.long, device=device)
        reply = language_model(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
        past = getattr(reply, 'past_key_values', None)
        output = []
        for i in range(max_new_tokens):
            chosen = torch.argmax(reply.logits[:, -1, :], dim=-1, keepdim=True)
            token = int(chosen.item())
            if tokenizer.eos_token_id is not None and token == tokenizer.eos_token_id:
                break
            output.append(token)
            mask = torch.cat([mask, torch.ones((1,1), dtype=torch.long, device=device)], dim=1)
            if past is not None:
                reply = language_model(input_ids=chosen, attention_mask=mask,
                                       past_key_values=past, use_cache=True)
                past = getattr(reply, 'past_key_values', None)
            else:
                
                new_embed = language_model.get_input_embeddings()(chosen)
                embeds = torch.cat([embeds, new_embed], dim=1)
                reply = language_model(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
    return tokenizer.decode(output, skip_special_tokens=True)


def iter_examples(manifest_path: Path, project_root: Path, split: str = 'vision_train'):
    from shards_io import resolve_shards
    if split not in ('vision_train', 'vision_validation'):
        raise ValueError('Invalid split')
    allowed = (project_root / 'training_data').resolve()
    for path in resolve_shards(manifest_path, split):
        with path.open(encoding='utf-8') as src:
            for line in src:
                if not line.strip():
                    continue
                record = json.loads(line)
                image_path = verify_vision_record(record, project_root)
                yield image_path, record
