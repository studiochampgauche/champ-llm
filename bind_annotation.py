#!/usr/bin/env python3
"""Explicitly bind a reviewed image to its annotation and SHA-256 hash."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from vision_integrity import sha256_file, utc_timestamp


def bind(image: Path, *, base: Path, confirm: bool = False) -> Path:
    image = Path(image).resolve()
    base = Path(base).resolve()
    if not image.is_relative_to(base) or image.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
        raise ValueError('Image must be stored under training_data/')
    if not image.is_file() or image.is_symlink():
        raise ValueError('Image is missing or is a disallowed symlink')
    annotation = image.with_suffix('.vision.json')
    legacy = image.with_suffix('.caption.txt')
    if annotation.is_file():
        if annotation.is_symlink():
            raise ValueError('Annotation symlinks are not allowed')
        data = json.loads(annotation.read_text(encoding='utf-8-sig'))
    elif legacy.is_file() and not legacy.is_symlink():
        data = {'description': legacy.read_text(encoding='utf-8-sig'), 'critique': '',
                'question': 'Describe this web interface.', 'code_files': []}
    else:
        data = {'description': '', 'critique': '', 'improvements': '', 'correction': '',
                'question': 'Analyze this web interface.', 'code_files': []}
    if not isinstance(data, dict):
        raise ValueError('Invalid annotation')
    digest = sha256_file(image)
    previous = data.get('image_sha256')
    if previous and previous != digest:
        raise ValueError('This annotation belongs to a DIFFERENT image. Do not rebind it automatically. '
                         'Create a new annotation after reviewing the new screenshot.')
    if not confirm:
        raise ValueError(f'Before binding, open and review this image: {image}\n'
                         'Then rerun with --confirm if the description and critique are accurate.')
    data['image_sha256'] = digest
    data['liaison_confirmee_utc'] = utc_timestamp()
    annotation.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    return annotation


def main():
    ap = argparse.ArgumentParser(description='Bind an existing image to its annotation after human review.')
    ap.add_argument('--image', type=Path, required=True)
    ap.add_argument('--confirm', action='store_true', help='Confirm that you reviewed this annotation against THIS image')
    args = ap.parse_args()
    base = Path(__file__).resolve().parent / 'training_data'
    try:
        result = bind(args.image, base=base, confirm=args.confirm)
    except ValueError as exc:
        ap.error(str(exc))
    print('Annotation verified and bound :', result)

if __name__ == '__main__':
    main()
