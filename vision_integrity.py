#!/usr/bin/env python3
"""Verify that images match the SHA-256 hashes in their annotations."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_capture(dest: Path, base_name: str, content: bytes) -> tuple[Path, str]:
    """Implement save capture for the local model pipeline."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content).hexdigest()
    image = dest / f'{base_name}-{digest[:16]}.png'
    if image.exists():
        if sha256_file(image) != digest:
            raise ValueError('Image hash collision or modified screenshot : ' + str(image))
        return image, digest
    fd, tmp = tempfile.mkstemp(prefix='.capture-', suffix='.png', dir=dest)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        if image.exists():
            if sha256_file(image) != digest:
                raise ValueError('Image hash collision or modified screenshot : ' + str(image))
        else:
            os.replace(tmp, image)
        return image, digest
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def verify_image_binding(image: Path, annotation: Path, *, expected_hash: str | None = None,
                     annotation_sha256: str | None = None) -> str:
    """Handle visual data or model processing: verify image binding."""
    image, annotation = Path(image), Path(annotation)
    if not annotation.is_file() or annotation.is_symlink() or image.is_symlink():
        raise ValueError('Image or annotation missing, or symlinks are disallowed')
    raw = annotation.read_bytes()
    if annotation_sha256 and hashlib.sha256(raw).hexdigest() != annotation_sha256:
        raise ValueError('Annotation changed since preparation: rebuild vision shards')
    obj = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError('Invalid annotation JSON')
    claimed = obj.get('image_sha256')
    if not isinstance(claimed, str) or len(claimed) != 64 or any(c not in '0123456789abcdef' for c in claimed):
        raise ValueError('Annotation not bound to a verified image (image_sha256 missing/invalid). '
                         'Review this screenshot before using bind_annotation.py --confirm.')
    real = sha256_file(image)
    if claimed != real or (expected_hash and expected_hash != real):
        raise ValueError('Image changed; annotation no longer matches its SHA-256')
    return real


def verify_vision_record(record: dict, project: Path) -> Path:
    """Handle visual data or model processing: verify vision record."""
    directory = (Path(project) / 'training_data').resolve()
    image = (Path(project) / record['image']).resolve()
    if not image.is_relative_to(directory) or not image.is_file() or image.is_symlink():
        raise ValueError('Image is missing or outside training_data : ' + str(image))
    annotation = image.with_suffix('.vision.json')
    expected_annotation = record.get('annotation')
    if not expected_annotation or (Path(project) / expected_annotation).resolve() != annotation:
        raise ValueError('Annotation missing or not associated with this capture')
    if not annotation.is_relative_to(directory):
        raise ValueError('Annotation is outside the allowed directory')
    expected = record.get('image_sha256')
    if not expected or not record.get('annotation_sha256'):
        raise ValueError('Old shards lack hashes: rerun prepare_vision.py')
    verify_image_binding(image, annotation, expected_hash=expected,
                     annotation_sha256=record['annotation_sha256'])
    return image


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
