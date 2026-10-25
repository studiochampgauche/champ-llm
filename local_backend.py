#!/usr/bin/env python3
"""Load local champ-llm weights for coding and optionally vision."""
from __future__ import annotations

import json
from pathlib import Path


def resolve_vision_checkpoint(folder: Path) -> Path:
    """Handle resumable training state: resolve vision checkpoint."""
    folder = Path(folder).expanduser().resolve()
    if (folder / 'latest.json').is_file():
        from vision_checkpoints import checkpoint_path
        folder = checkpoint_path(folder, resume=True)
    if not (folder / 'vision_config.json').is_file() or not (folder / 'vision.safetensors').is_file():
        raise FileNotFoundError(f'Incomplete vision checkpoint: {folder}')
    return folder


def verify_local_model(folder: Path):
    folder = Path(folder).expanduser().resolve()
    if not (folder / 'config.json').is_file():
        raise FileNotFoundError(f'Model configuration not found : {folder / "config.json"}')
    if not (folder / 'tokenizer.json').is_file():
        raise FileNotFoundError(f'Local tokenizer not found : {folder / "tokenizer.json"}')
    if not any(folder.glob('*.safetensors')):
        raise FileNotFoundError(f'safetensors weights not found in {folder}')
    return folder


class ChampBackend:
    """ChampBackend provides reusable operations for this project."""

    def __init__(self, model: Path, *, vision: Path | None = None,
                 device: str = 'auto', max_generation: int = 512,
                 max_vision_generation: int = 180):
        if max_generation < 1 or max_vision_generation < 1:
            raise ValueError('Generation length must be positive')
        if device not in ('auto', 'cpu', 'cuda'):
            raise ValueError('Expected device: auto, cpu or cuda')
        checkpoint = resolve_vision_checkpoint(vision) if vision is not None else None
        model = verify_local_model(model)
        if checkpoint is not None:
            meta = json.loads((checkpoint/'metadata.json').read_text(encoding='utf-8')) if (checkpoint/'metadata.json').is_file() else {}
            if meta.get('mode') == 'joint':
                source = checkpoint/'language'
                if source.is_dir() and model != source.resolve():
                    raise ValueError('Joint visual training: use --local-model ' + str(source) +
                                     ' (text weights from the same checkpoint), not an older version of the LLM.')
                if not source.is_dir():
                    raise ValueError('Joint checkpoint has no language/ directory')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = self._select_device(torch, device, model)
        self.dtype = (torch.bfloat16 if self.device.type == 'cuda' and torch.cuda.is_bf16_supported()
                      else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True,
                                                        trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model), local_files_only=True, trust_remote_code=False,
            torch_dtype=self.dtype, low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.max_generation = max_generation
        self.max_vision_generation = max_vision_generation
        self.vision = None
        self.vision_checkpoint = checkpoint
        if checkpoint is not None:
            from safetensors.torch import load_file
            from vision_model import VisionEncoder
            cfg = json.loads((checkpoint/'vision_config.json').read_text(encoding='utf-8'))
            if cfg.get('version') != 2 or cfg.get('llm_hidden') != self.model.config.hidden_size:
                raise ValueError('Vision adapter architecture is incompatible with the local LLM')
            bridge = VisionEncoder(
                cfg['llm_hidden'], image_size=cfg['image_size'], patch_size=cfg['patch_size'],
                vision_dim=cfg['vision_dim'], vision_layers=cfg['vision_layers'],
                max_tiles=cfg['max_tiles'], pool_grid=cfg['pool_grid'],
            )
            bridge.load_state_dict(load_file(str(checkpoint/'vision.safetensors')))
            self.vision = bridge.to(self.device).eval()
        self.model_dir = model

    @staticmethod
    def _select_device(torch, device: str, folder: Path):
        if device == 'cpu':
            return torch.device('cpu')
        if device == 'cuda':
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but not available')
            return torch.device('cuda')
        if not torch.cuda.is_available():
            return torch.device('cpu')
        weights = sum(p.stat().st_size for p in folder.glob('*.safetensors'))
        try:
            free, total = torch.cuda.mem_get_info()
        except RuntimeError:
            return torch.device('cpu')
        
        if weights * 1.25 > free:
            print(f'Warning: weights ~{weights / 1e9:.1f} GB, free VRAM ~{free / 1e9:.1f} GB. '
                  'Falling back to CPU to avoid CUDA OOM. Large-model inference may be very slow.', flush=True)
            return torch.device('cpu')
        return torch.device('cuda')

    @property
    def has_vision(self) -> bool:
        return self.vision is not None

    def generate_code(self, prompt: str) -> str:
        """Implement generate code for the local model pipeline."""
        import torch
        from text_training_common import assistant_prompt_tokens
        
        # BOS + <|user|>\n...\n<|end|>\n<|assistant|>\n
        
        
        ids = torch.tensor([assistant_prompt_tokens(self.tokenizer, prompt)], dtype=torch.long)
        context = getattr(self.model.config, 'max_position_embeddings', None)
        if context and ids.shape[1] + self.max_generation > context:
            raise ValueError(f'Prompt too long ({ids.shape[1]} tokens + '
                             f'{self.max_generation} generated tokens exceed context {context}). '
                             'Reduce --max-source-chars or --max-generation; '
                             'source code will not be silently truncated.')
        ids = ids.to(self.device)
        opts = {'max_new_tokens': self.max_generation, 'do_sample': False}
        
        
        stop_ids = []
        if self.tokenizer.eos_token_id is not None:
            stop_ids.append(self.tokenizer.eos_token_id)
        vocabulary = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else {}
        end_id = vocabulary.get('<|end|>') if isinstance(vocabulary, dict) else None
        if isinstance(end_id, int) and end_id >= 0 and end_id not in stop_ids:
            stop_ids.append(end_id)
        if stop_ids:
            opts['eos_token_id'] = stop_ids[0] if len(stop_ids) == 1 else stop_ids
        if self.tokenizer.pad_token_id is not None:
            opts['pad_token_id'] = self.tokenizer.pad_token_id
        with torch.inference_mode():
            generated = self.model.generate(input_ids=ids, **opts)
        return self.tokenizer.decode(generated[0, ids.shape[1]:], skip_special_tokens=True).strip()

    def critique_images(self, images: list[Path]) -> str:
        if not self.has_vision:
            return ('Local vision unavailable: provide --vision-checkpoint with '
                    'trained vision weights. No Ollama model is used.')
        from vision_model import generate_answer, image_to_tiles
        cfg = self.vision.vision_config
        outputs = []
        for path in images[:2]:
            pixels = image_to_tiles(path, cfg['image_size'], cfg['max_tiles'])
            pixels = pixels.unsqueeze(0).to(self.device)
            question = (
                'Review this website screenshot. Identify specific problems with alignment, '
                'contrast, readability, responsiveness, and visual hierarchy. '
                'Suggest precise CSS improvements without inventing measurements. '
                'You may respond in French when the user requests French.'
            )
            critique = generate_answer(self.model, self.vision, self.tokenizer, pixels,
                                      question, max_new_tokens=self.max_vision_generation)
            outputs.append(f'{path.name} : {critique[:4000]}')
        return '\n'.join(outputs)[:8000]


# Compatibility for Python integrations using the original class name.
ChampGaucheBackend = ChampBackend
