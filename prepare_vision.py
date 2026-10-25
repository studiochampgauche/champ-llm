#!/usr/bin/env python3
"""Prepare image, description, critique and code associations for training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from vision_integrity import verify_image_binding, sha256_file

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}
SKIP_DIRS = {'.git', '.venv', 'node_modules', 'vendor', 'dist', 'build', '__pycache__', '.next', 'outputs'}
SKIP_NAMES = {'.env', 'wp-config.php', 'secrets.json', 'credentials.json'}
SAFE_CODE_EXTS = {'.js', '.jsx', '.ts', '.tsx', '.php', '.html', '.css', '.scss', '.sass', '.py', '.md', '.json', '.vue', '.svelte', '.astro', '.sql', '.svg', '.txt'}


def hash_split(project: str, validation_ratio: float = .1) -> str:
    """Implement hash split for the local model pipeline."""
    h = hashlib.sha256(project.encode('utf-8')).digest()
    x = int.from_bytes(h[:8], 'big') / 2**64
    return 'vision_validation' if x < validation_ratio else 'vision_train'


def safe_relative(path_text: str, base: Path) -> Path:
    """Implement safe relative for the local model pipeline."""
    candidate = Path(path_text)
    if candidate.is_absolute() or '..' in candidate.parts or not candidate.parts:
        raise ValueError('Absolute paths and traversal are not allowed: ' + path_text)
    full = (base / candidate).resolve()
    if not full.is_relative_to(base.resolve()):
        raise ValueError('Path escapes training_data: ' + path_text)
    if any(item.is_symlink() for item in (base / candidate).parents if item != base.parent):
        raise ValueError('Symlinks are not allowed')
    if full.is_symlink():
        raise ValueError('Symlinks are not allowed')
    return full


def iter_images(root: Path):
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in SKIP_DIRS and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in IMAGE_EXTS and not path.is_symlink():
                yield path


def load_code_files(base: Path, paths: list[str], max_each: int = 32_768) -> tuple[list[dict], list[str]]:
    snippets, warnings = [], []
    for rel in paths[:12]:
        try:
            src = safe_relative(str(rel), base)
            if src.suffix.lower() not in SAFE_CODE_EXTS or src.name.lower() in SKIP_NAMES or src.name.lower().startswith('.env'):
                raise ValueError('disallowed extension')
            if not src.is_file() or src.stat().st_size > max_each:
                raise ValueError('File is missing or too large')
            content = src.read_text(encoding='utf-8-sig')
            if re.search(r'(?i)(-----BEGIN .+PRIVATE KEY|(?:password|secret|api_key|private_key)\s*[:=])', content):
                raise ValueError('Potential secret in source code')
            snippets.append({'path': str(rel).replace('\\', '/'), 'content': content})
        except (OSError, UnicodeError, ValueError) as exc:
            warnings.append(f'Code excluded {rel}: {exc}')
    return snippets, warnings


def build_answer(annotation: dict, code: list[dict]) -> str:
    sections = []
    for key, label in [('description', 'Visual description'), ('critique', 'Problems and critique'),
                       ('improvements', 'Recommended improvements'), ('correction', 'Correction')]:
        val = annotation.get(key, '')
        if isinstance(val, str) and val.strip():
            sections.append(f'{label}:\n{val.strip()}')
    for snippet in code:
        sections.append(f"Associated code ({snippet['path']}):\n{snippet['content']}")
    return '\n\n'.join(sections)


def prepare_vision(sources: Path, outputs: Path, *, validation: float = .1, shard_mb: float = 16.0,
                   root: Path | None = None) -> dict:
    if not 0 <= validation < .5 or shard_mb <= 0:
        raise ValueError('Require 0 <= validation < 0.5 and shard_mb > 0')
    root = root or sources.parent
    outputs.mkdir(parents=True, exist_ok=True)
    vision_root = outputs / 'vision'
    max_shard_bytes = max(4096, int(shard_mb * 1024**2))
    from prepare_data import ShardWriter
    skipped, unannotated, accepted, warnings = [], [], 0, []
    with tempfile.TemporaryDirectory(prefix='.vision_', dir=outputs) as temp:
        stage = Path(temp)
        writers = {name: ShardWriter(stage, name, max_shard_bytes)
                   for name in ('vision_train', 'vision_validation')}
        try:
            for image_path in iter_images(sources):
                image_rel = image_path.relative_to(root).as_posix()
                sidecar = image_path.with_suffix('.vision.json')
                if not sidecar.is_file():
                    if image_path.with_suffix('.caption.txt').exists():
                        skipped.append({'image': image_rel, 'reason': 'Legacy .caption.txt without SHA-256 binding: review it and use bind_annotation.py --confirm'})
                    else:
                        unannotated.append(image_rel)
                    continue
                try:
                    
                    from PIL import Image
                    if image_path.stat().st_size > 25 * 1024**2:
                        raise ValueError('Image exceeds 25 MiB')
                    with Image.open(image_path) as photo:
                        photo.verify()
                    if sidecar.stat().st_size > 131072 or sidecar.is_symlink():
                        raise ValueError('Annotation is too large or is a symlink')
                    image_sha = verify_image_binding(image_path, sidecar)
                    annotation_sha = sha256_file(sidecar)
                    annotation = json.loads(sidecar.read_text(encoding='utf-8-sig'))
                    if not isinstance(annotation, dict):
                        raise ValueError('Annotation must be a JSON object')
                    project = annotation.get('project') or image_path.parent.relative_to(sources).as_posix()
                    if not isinstance(project, str) or not project.strip():
                        project = image_path.parent.relative_to(sources).as_posix()
                    code_paths = annotation.get('code_files', [])
                    if not isinstance(code_paths, list) or any(not isinstance(x, str) for x in code_paths):
                        raise ValueError('code_files must be a list of relative paths')
                    code, code_warnings = load_code_files(sources, code_paths)
                    warnings.extend(code_warnings)
                    answer = build_answer(annotation, code)
                    if not answer.strip():
                        unannotated.append(image_rel)
                        continue
                    question = annotation.get('question') or 'Analyze this interface and suggest appropriate improvements.'
                    if not isinstance(question, str):
                        raise ValueError('question must be a string')
                    record = {'image': image_rel, 'image_sha256': image_sha,
                              'annotation_sha256': annotation_sha, 'project': project, 'question': question,
                              'answer': answer, 'annotation': sidecar.relative_to(root).as_posix(),
                              'viewport_width': annotation.get('viewport_width'),
                              'phase': annotation.get('phase', 'example')}
                    split = hash_split(project, validation)
                    writers[split].write(json.dumps(record, ensure_ascii=False) + '\n')
                    accepted += 1
                except (ValueError, OSError, UnicodeError, json.JSONDecodeError) as exc:
                    skipped.append({'image': image_rel, 'reason': str(exc)})
        finally:
            for writer in writers.values():
                writer.close()
        manifest = {'version': 1, 'integrity_version': 2, 'format': 'champ-llm image + critique + code, SHA-256 bound',
                    'splits': {key: value.summary() for key, value in writers.items()},
                    'note': 'Shard paths are relative to outputs/vision; images to the project root.'}
        (stage / 'vision_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        report = {'annotated_images': accepted, 'unannotated_images': unannotated,
                  'skipped_images': skipped, 'warnings': warnings,
                  'note': 'Vision dataset prepared; vision weights were not trained.'}
        (stage / 'vision_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        
        old = vision_root.with_name('vision_old_temp')
        if old.exists():
            import shutil
            shutil.rmtree(old)
        if vision_root.exists():
            os.replace(vision_root, old)
        try:
            os.replace(stage, vision_root)
        except BaseException:
            if old.exists() and not vision_root.exists():
                os.replace(old, vision_root)
            raise
        if old.exists():
            import shutil
            shutil.rmtree(old)
    return report


def main():
    p = argparse.ArgumentParser(description='Prepare images, descriptions, critiques and code without training.')
    p.add_argument('--validation', type=float, default=.1)
    p.add_argument('--shard-mb', type=float, default=16)
    a = p.parse_args()
    base = Path(__file__).resolve().parent
    report = prepare_vision(base/'training_data', base/'outputs', validation=a.validation, shard_mb=a.shard_mb, root=base)
    print('Vision :', report['annotated_images'], 'annotated example(s),',
          len(report['unannotated_images']), 'screenshot(s) without annotations.')
    print('Manifest :', base/'outputs'/'vision'/'vision_manifest.json')
    print('No model weights were modified.')


if __name__ == '__main__':
    main()
