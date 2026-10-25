#!/usr/bin/env python3
"""Stream, validate, de-duplicate and shard text/code/conversation training data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / 'training_data'
SINGLE = ROOT / 'training_data.txt'
OUTPUT = ROOT / 'outputs'

IGNORED_DIRS = {
    '.git', '.venv', 'venv', 'node_modules', 'vendor', 'dist', 'build', '.next', '.nuxt',
    '.svelte-kit', '.astro', 'coverage', '__pycache__', '.idea', '.cache',
    '.turbo', 'target', 'tmp', 'temp', 'bin', 'obj', '.output', 'out',
}
IGNORED_NAMES = {
    'wp-config.php', 'credentials.json', 'secrets.json', 'auth.json', '.npmrc',
    '.pypirc', '.netrc', 'id_rsa', 'id_ed25519', 'composer.lock', 'package-lock.json',
    'pnpm-lock.yaml', 'yarn.lock', 'bun.lockb',
}
BLOCKED_SUFFIXES = {
    '.pem', '.key', '.p12', '.pfx', '.p8', '.jks', '.sqlite', '.sqlite3', '.db',
    '.zip', '.7z', '.rar', '.tar', '.gz', '.exe', '.dll', '.so', '.dylib', '.pyc',
    '.pdf', '.docx', '.woff', '.woff2', '.ttf', '.otf', '.mp3', '.mp4', '.wav',
    '.webm', '.ico', '.gif', '.webp', '.jpeg', '.jpg', '.png', '.bmp', '.avif',
    '.lock', '.map', '.bin', '.ckpt', '.pt', '.pth', '.safetensors', '.gguf',
}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.avif'}
LANGUAGE = {
    '.js': 'JavaScript', '.mjs': 'JavaScript', '.cjs': 'JavaScript',
    '.ts': 'TypeScript', '.mts': 'TypeScript', '.cts': 'TypeScript',
    '.jsx': 'React JSX', '.tsx': 'React TSX', '.php': 'PHP',
    '.css': 'CSS', '.sass': 'Sass', '.scss': 'SCSS', '.less': 'Less',
    '.html': 'HTML', '.htm': 'HTML', '.vue': 'Vue', '.svelte': 'Svelte',
    '.astro': 'Astro', '.mdx': 'MDX', '.twig': 'Twig', '.blade.php': 'Blade PHP',
    '.py': 'Python', '.sql': 'SQL', '.graphql': 'GraphQL', '.gql': 'GraphQL',
    '.prisma': 'Prisma', '.json': 'JSON', '.jsonc': 'JSONC', '.xml': 'XML',
    '.yaml': 'YAML', '.yml': 'YAML', '.toml': 'TOML', '.go': 'Go', '.rs': 'Rust',
    '.sh': 'Shell', '.bash': 'Bash', '.ps1': 'PowerShell', '.java': 'Java',
    '.kt': 'Kotlin', '.cs': 'C#', '.rb': 'Ruby', '.swift': 'Swift',
    '.c': 'C', '.h': 'C/C++', '.cpp': 'C++', '.hpp': 'C++',
    '.txt': 'Text', '.md': 'Markdown', '.jsonl': 'JSONL', '.svg': 'SVG',
}
SECRET_DETECTORS = [
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    re.compile(r'\b(?:ghp_|gho_|ghu_|ghs_|github_pat_|sk_live_|rk_live_|sk-proj-)[A-Za-z0-9_-]{12,}'),
    re.compile(r'(?im)^\s*(?:export\s+)?(?:[A-Z_]*(?:PASSWORD|API_KEY|SECRET_KEY|ACCESS_TOKEN|PRIVATE_KEY|DATABASE_URL|DB_PASSWORD))\s*[:=]\s*[\'\"]?[^\s\'\"#]{8,}'),
]
IMPORT_RE = re.compile(
    r'(?:\b(?:from|import|require)\s*\(?\s*[\'\"](\.[^\'\"]+)[\'\"]|'
    r'\b(?:@use|@forward|@import)\s*[\'\"](\.[^\'\"]+)[\'\"])'
)
SPLIT_RE = re.compile(r'^[ \t]*===[ \t]*$')
VALID_ROLES = {'system', 'user', 'assistant', 'tool'}
READ_CHARS = 65536
MAX_JSONL_RECORD_CHARS_DEFAULT = 8 * 1024 * 1024


def file_language(path: Path) -> str:
    if path.name.lower().endswith('.blade.php'):
        return 'Blade PHP'
    return LANGUAGE.get(path.suffix.lower(), 'Text / code (custom extension)')


def find_sources():
    if SINGLE.is_file():
        yield SINGLE
    if SOURCES.is_dir():
        for parent, dirs, files in os.walk(SOURCES, topdown=True, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d.lower() not in IGNORED_DIRS
                             and not (Path(parent) / d).is_symlink())
            for filename in sorted(files):
                yield Path(parent) / filename


def safe_source(path: Path, max_bytes: int) -> tuple[bool, str]:
    name = path.name.lower()
    if path.is_symlink() or not path.is_file():
        return False, 'symlink or non-regular file'
    if (name.startswith('.env') or name in IGNORED_NAMES
            or name.endswith(('.min.js', '.min.css', '.vision.json', '.caption.txt'))):
        return False, 'sensitive or generated file'
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False, 'binary or unsupported non-text file'
    if path.stat().st_size > max_bytes:
        return False, 'file too large'
    return True, ''


def precheck_stream(path: Path) -> tuple[bool, str, str]:
    """Handle streaming training shards: precheck stream."""
    sha = hashlib.sha256()
    previous = ''
    n_chars = 0
    controls = 0
    try:
        with path.open('r', encoding='utf-8-sig', errors='strict', newline='') as f:
            while True:
                block = f.read(READ_CHARS)
                if not block:
                    break
                sha.update(block.encode('utf-8'))
                if '\x00' in block:
                    return False, 'binary data (NUL byte)', ''
                n_chars += len(block)
                controls += sum(ord(c) < 32 and c not in '\n\r\t\f' for c in block)
                sample = previous + block
                if any(rx.search(sample) for rx in SECRET_DETECTORS):
                    return False, 'potential secret detected (review required)', ''
                previous = sample[-4096:]
        if n_chars and controls / n_chars >= .002:
            return False, 'non-text control characters', ''
        return True, '', sha.hexdigest()
    except (UnicodeError, OSError) as exc:
        return False, f'Invalid UTF-8 or read error: {exc}', ''


def iter_bounded_lines(path: Path, piece_chars: int = READ_CHARS):
    """Implement iter bounded lines for the local model pipeline."""
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        buffer = ''
        while True:
            incoming = stream.read(piece_chars)
            if not incoming:
                if buffer:
                    yield buffer
                return
            data = buffer + incoming
            offset = 0
            while True:
                pos = data.find('\n', offset)
                if pos < 0:
                    break
                yield data[offset:pos + 1]
                offset = pos + 1
            buffer = data[offset:]
            while len(buffer) >= piece_chars:
                yield buffer[:piece_chars]
                buffer = buffer[piece_chars:]


def chunks_by_chars(text: str, max_chars: int):
    """Implement chunks by chars for the local model pipeline."""
    if max_chars <= 0:
        raise ValueError('max_chars must be positive')
    remaining = text
    while len(remaining) > max_chars:
        cut = remaining.rfind('\n', max_chars // 2, max_chars)
        end = cut + 1 if cut >= 0 else max_chars
        piece, remaining = remaining[:end], remaining[end:]
        if piece.strip():
            yield piece
    if remaining.strip():
        yield remaining


def iter_file_segments(path: Path, max_chars: int, section_split: bool):
    """Implement iter file segments for the local model pipeline."""
    buffer = ''
    section = 0
    segment = 0
    for line in iter_bounded_lines(path):
        if section_split and SPLIT_RE.fullmatch(line.rstrip('\r\n')):
            if buffer.strip():
                segment += 1
                yield section, segment, buffer
            buffer = ''
            section += 1
            segment = 0
            continue
        buffer += line
        while len(buffer) >= max_chars:
            cut = buffer.rfind('\n', max_chars // 2, max_chars)
            end = cut + 1 if cut >= 0 else max_chars
            piece, buffer = buffer[:end], buffer[end:]
            if piece.strip():
                segment += 1
                yield section, segment, piece
    if buffer.strip():
        segment += 1
        yield section, segment, buffer


def iter_jsonl_records(path: Path, max_chars: int):
    """Implement iter jsonl records for the local model pipeline."""
    parts = []
    size = 0
    too_large = False
    line_number = 0
    for piece in iter_bounded_lines(path):
        if not too_large:
            size += len(piece.encode('utf-8'))
            if size > max_chars:
                too_large = True
                parts.clear()
            else:
                parts.append(piece)
        if piece.endswith('\n'):
            line_number += 1
            yield line_number, None if too_large else ''.join(parts)
            parts.clear()
            too_large = False
            size = 0
    if parts or too_large:
        yield line_number + 1, None if too_large else ''.join(parts)


def conversations_to_text(messages: list[dict]) -> str:
    return '\n'.join(
        f"<|{message['role']}|>\n{message['content']}\n<|end|>" for message in messages
    )


def valid_messages(messages) -> bool:
    return (isinstance(messages, list) and len(messages) >= 2
            and all(isinstance(m, dict) and m.get('role') in VALID_ROLES
                    and isinstance(m.get('content'), str) and m['content'].strip()
                    for m in messages)
            and any(m['role'] == 'assistant' for m in messages)
            and any(m['role'] == 'user' for m in messages))


def stable_validation(identifier: str, ratio: float) -> bool:
    h = hashlib.sha256(identifier.encode('utf-8')).digest()
    return int.from_bytes(h[:8], 'big') / (2 ** 64) < ratio


def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(',', ':'))


class ShardWriter:
    def __init__(self, root: Path, split: str, max_bytes: int, suffix: str = '.jsonl'):
        self.rel_dir = Path('shards') / split
        self.folder = root / self.rel_dir
        self.folder.mkdir(parents=True, exist_ok=True)
        self.max_bytes = max_bytes
        self.suffix = suffix
        self.fp = None
        self.current_size = 0
        self.current_count = 0
        self.count = 0
        self.files = []

    def _open(self):
        index = len(self.files)
        rel = self.rel_dir / f'shard-{index:05d}{self.suffix}'
        self.fp = (self.folder / rel.name).open('w', encoding='utf-8', newline='\n')
        self.files.append({'path': rel.as_posix(), 'records': 0, 'bytes': 0})
        self.current_size = self.current_count = 0

    def write(self, record: str):
        data = record.encode('utf-8')
        if self.fp and self.current_size > 0 and self.current_size + len(data) > self.max_bytes:
            self.fp.close()
            self.fp = None
        if not self.fp:
            self._open()
        self.fp.write(record)
        self.current_size += len(data)
        self.current_count += 1
        self.count += 1
        self.files[-1]['records'] = self.current_count
        self.files[-1]['bytes'] = self.current_size

    def close(self):
        if self.fp:
            self.fp.close()
            self.fp = None

    def summary(self):
        return {'records': self.count, 'files': self.files}


def write_jsonl(path: Path, cursor):
    with path.open('w', encoding='utf-8', newline='\n') as output:
        for (payload,) in cursor:
            output.write(payload + '\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Streaming UTF-8 preparation with sharding and document-level splits.')
    parser.add_argument('--max-mb', type=float, default=64.0,
                        help='Maximum UTF-8 source size in MiB (default 64).')
    parser.add_argument('--chunk-chars', type=int, default=8192,
                        help='Segment length in characters (default 8192).')
    parser.add_argument('--shard-mb', type=float, default=16.0,
                        help='Target shard size in MiB (default 16).')
    parser.add_argument('--max-record-mb', type=float, default=8.0,
                        help='Maximum JSONL record size in MiB (default 8).')
    parser.add_argument('--validation', type=float, default=.05,
                        help='Fraction of documents reserved for validation (default 0.05).')
    args = parser.parse_args(argv)
    if not 0 < args.validation < .5:
        parser.error('--validation must be greater than 0 and less than 0.5')
    if not 512 <= args.chunk_chars <= 2_000_000:
        parser.error('--chunk-chars must be between 512 and 2000000')
    if min(args.max_mb, args.shard_mb, args.max_record_mb) <= 0:
        parser.error('--max-mb, --shard-mb and --max-record-mb must be positive')

    max_bytes = int(args.max_mb * 1024**2)
    max_record_chars = int(args.max_record_mb * 1024**2)
    shard_bytes = max(128, int(args.shard_mb * 1024**2))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    processed = total_bytes = image_count = skipped_count = 0
    with tempfile.TemporaryDirectory(dir=OUTPUT, prefix='.preparation_') as work:
        stage = Path(work)
        db = sqlite3.connect(str(stage / 'data.sqlite3'))
        try:
            db.execute('PRAGMA journal_mode=DELETE')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('PRAGMA cache_size=-8192')
            db.executescript('''
                CREATE TABLE pretrain(
                    digest TEXT PRIMARY KEY, group_id TEXT NOT NULL, conversation_id TEXT,
                    payload TEXT NOT NULL
                );
                CREATE INDEX pretrain_group ON pretrain(group_id);
                CREATE TABLE conversations(digest TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE TABLE file_inventory(payload TEXT NOT NULL);
                CREATE TABLE images(payload TEXT NOT NULL);
                CREATE TABLE skipped_files(payload TEXT NOT NULL);
            ''')

            def ignore(rel, reason):
                nonlocal skipped_count
                skipped_count += 1
                db.execute('INSERT INTO skipped_files(payload) VALUES (?)',
                           (to_json({'file': rel, 'reason': reason}),))

            def add_pretrain(content, rendered=None, group=None, conversation_id=None):
                digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
                group_id = group or ('record:' + digest)
                db.execute('INSERT OR IGNORE INTO pretrain VALUES (?,?,?,?)',
                           (digest, group_id, conversation_id, to_json({'text': content if rendered is None else rendered})))
                if conversation_id:
                    db.execute('UPDATE pretrain SET conversation_id=? WHERE digest=? AND conversation_id IS NULL',
                               (conversation_id, digest))

            for path in find_sources():
                try:
                    rel = path.relative_to(ROOT).as_posix()
                    if path.suffix.lower() in IMAGE_SUFFIXES:
                        if not path.is_symlink() and path.stat().st_size <= 25 * 1024**2:
                            sidecar = path.with_suffix('.caption.txt')
                            caption = ''
                            if sidecar.is_file() and not sidecar.is_symlink() and sidecar.stat().st_size <= 16384:
                                caption = sidecar.read_text(encoding='utf-8-sig').strip()
                            db.execute('INSERT INTO images VALUES (?)', (to_json({
                                'image': rel, 'caption': caption, 'annotation_complete': bool(caption)
                            }),))
                            image_count += 1
                        continue
                    allowed, reason = safe_source(path, max_bytes)
                    if not allowed:
                        ignore(rel, reason)
                        continue
                    valid, reason, file_hash = precheck_stream(path)
                    if not valid:
                        ignore(rel, reason)
                        continue
                except (OSError, UnicodeError) as exc:
                    ignore(str(path), f'could not read: {exc}')
                    continue

                lang = file_language(path)
                size_bytes = path.stat().st_size
                imports = set()
                previous_import_text = ''
                savepoint = f'file_{processed}'
                db.execute(f'SAVEPOINT {savepoint}')
                try:
                    if path.suffix.lower() == '.jsonl':
                        for line_no, raw in iter_jsonl_records(path, max_record_chars):
                            if raw is None:
                                ignore(rel, f'line {line_no}: JSONL record too large')
                                continue
                            if not raw.strip():
                                continue
                            try:
                                obj = json.loads(raw)
                            except json.JSONDecodeError:
                                ignore(rel, f'line {line_no}: invalid JSONL')
                                continue
                            if isinstance(obj, dict) and isinstance(obj.get('messages'), list):
                                if any(isinstance(message, dict) and 'apprendre' in message for message in obj['messages']):
                                    raise ValueError('Legacy conversation field apprendre detected: rename it to learn before training.')
                            if isinstance(obj, dict) and valid_messages(obj.get('messages')):
                                messages = [{'role': m['role'], 'content': m['content'], **({'learn': m['learn']} if m['role'] == 'assistant' and isinstance(m.get('learn'), bool) else {})} for m in obj['messages']]
                                canonical = json.dumps(messages, ensure_ascii=False, sort_keys=True)
                                conv_id = hashlib.sha256(canonical.encode('utf-8')).hexdigest()
                                db.execute('INSERT OR IGNORE INTO conversations VALUES (?,?)',
                                           (conv_id, to_json({'messages': messages})))
                                rendered = conversations_to_text(messages)
                                for part in chunks_by_chars(rendered, args.chunk_chars):
                                    add_pretrain(part, part, group='conversation:' + conv_id,
                                                 conversation_id=conv_id)
                            elif isinstance(obj, dict) and isinstance(obj.get('text'), str) and obj['text'].strip():
                                content = obj['text']
                                doc_id = hashlib.sha256(content.encode('utf-8')).hexdigest()
                                for part in chunks_by_chars(content, args.chunk_chars):
                                    add_pretrain(part, part, group='record:' + doc_id)
                            else:
                                ignore(rel, f'line {line_no}: expected text or messages')
                    else:
                        group_id = 'file:' + file_hash
                        segments = 0
                        for section, part_idx, piece in iter_file_segments(
                            path, args.chunk_chars,
                            path.suffix.lower() in {'.txt', '.md', '.mdx'} or path == SINGLE,
                        ):
                            segments += 1
                            if path == SINGLE:
                                rendered = piece
                            else:
                                header = (f'<|file_start|>\nPath: {rel}\nLanguage: {lang}\n'
                                          f'Extension: {path.suffix.lower() or "(aucune)"}\n'
                                          f'Section: {section + 1}\nPart: {part_idx}\nContent:\n')
                                rendered = header + piece + '\n<|file_end|>'
                            add_pretrain(piece, rendered, group_id)
                        
                        for part in iter_bounded_lines(path):
                            window = previous_import_text + part
                            for vals in IMPORT_RE.findall(window):
                                if len(imports) < 1000:
                                    imports.update(v for v in vals if v)
                            previous_import_text = window[-512:]
                    db.execute('INSERT INTO file_inventory VALUES (?)', (to_json({
                        'path': rel, 'extension': path.suffix.lower() or '(aucune)',
                        'language': lang, 'size_bytes': size_bytes,
                        'detected_relative_imports': sorted(imports),
                    }),))
                    db.execute(f'RELEASE SAVEPOINT {savepoint}')
                    processed += 1
                    total_bytes += size_bytes
                    if processed % 1000 == 0:
                        print(f'  {processed:,} files lus...', flush=True)
                except (OSError, UnicodeError) as exc:
                    db.execute(f'ROLLBACK TO SAVEPOINT {savepoint}')
                    db.execute(f'RELEASE SAVEPOINT {savepoint}')
                    ignore(rel, f'error while reading: {exc}')
            db.commit()
            pretrain_count = db.execute('SELECT COUNT(*) FROM pretrain').fetchone()[0]
            sft_count = db.execute('SELECT COUNT(*) FROM conversations').fetchone()[0]
            if not pretrain_count:
                raise SystemExit('No usable training data found. Add source files under training_data/.')

            def hashed_sft_val(conv_id):
                return sft_count >= 10 and stable_validation('conversation:' + conv_id, args.validation)

            conv_val = sum(hashed_sft_val(row[0]) for row in db.execute('SELECT digest FROM conversations'))
            force_conv_val = force_conv_train = None
            if sft_count >= 10:
                first = db.execute('SELECT digest FROM conversations ORDER BY rowid LIMIT 1').fetchone()[0]
                if conv_val == 0:
                    force_conv_val = first
                elif conv_val == sft_count:
                    force_conv_train = first

            def conversation_validation(conv_id):
                if conv_id == force_conv_val:
                    return True
                if conv_id == force_conv_train:
                    return False
                return hashed_sft_val(conv_id)

            group_count = db.execute('SELECT COUNT(DISTINCT group_id) FROM pretrain').fetchone()[0]
            def plain_val(group):
                return group_count >= 10 and stable_validation(group, args.validation)
            pretrain_val_count = sum(
                conversation_validation(conv) if conv else plain_val(group)
                for group, conv in db.execute('SELECT group_id,conversation_id FROM pretrain')
            )
            force_plain_val = force_plain_train = None
            if group_count >= 10 and pretrain_val_count in (0, pretrain_count):
                row = db.execute('SELECT group_id FROM pretrain WHERE conversation_id IS NULL ORDER BY rowid LIMIT 1').fetchone()
                if row:
                    if pretrain_val_count == 0:
                        force_plain_val = row[0]
                    else:
                        force_plain_train = row[0]

            def pretrain_validation(group, conv):
                if conv:
                    return conversation_validation(conv)
                if group == force_plain_val:
                    return True
                if group == force_plain_train:
                    return False
                return plain_val(group)

            writers = {
                split: ShardWriter(stage, split, shard_bytes)
                for split in ('pretrain_train', 'pretrain_validation',
                              'conversations_train', 'conversations_validation')
            }
            writers['tokenizer'] = ShardWriter(stage, 'tokenizer', shard_bytes, suffix='.txt')
            try:
                for group, conv, payload in db.execute('SELECT group_id,conversation_id,payload FROM pretrain ORDER BY rowid'):
                    dest = 'pretrain_validation' if pretrain_validation(group, conv) else 'pretrain_train'
                    writers[dest].write(payload + '\n')
                    writers['tokenizer'].write(json.loads(payload)['text'] + '\n\n')
                for digest, payload in db.execute('SELECT digest,payload FROM conversations ORDER BY rowid'):
                    dest = 'conversations_validation' if conversation_validation(digest) else 'conversations_train'
                    writers[dest].write(payload + '\n')
            finally:
                for writer in writers.values():
                    writer.close()

            manifest = {
                'version': 2,
                'format': 'champ-llm sharded JSONL v2',
                'splits': {split: writer.summary() for split, writer in writers.items()},
                'note': 'Paths are relative to outputs/. Tokenizer shards contain plain text.',
            }
            (stage / 'shards_manifest.json').write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            for name, table in (('file_inventory.jsonl', 'file_inventory'),
                                ('visual_manifest.jsonl', 'images')):
                write_jsonl(stage / name, db.execute(f'SELECT payload FROM {table} ORDER BY rowid'))

            report = {
                'source_files_read': processed,
                'source_bytes': total_bytes,
                'pretrain_documents': pretrain_count,
                'pretrain_training': writers['pretrain_train'].count,
                'pretrain_validation': writers['pretrain_validation'].count,
                'supervised_conversations': sft_count,
                'images_indexed_not_trained': image_count,
                'skipped_count': skipped_count,
                'parameters': {
                    'max_file_mib': args.max_mb, 'chunk_chars': args.chunk_chars,
                    'shard_mib': args.shard_mb, 'max_record_mib': args.max_record_mb,
                    'validation': args.validation,
                },
                'warning': ('Shards are not tokenized; the final context must be bounded in tokens. '
                              'Secret scanning is incomplete. Deduplication is exact; near-duplicates '
                              'between training and validation may remain.'),
            }
            with (stage / 'report.json').open('w', encoding='utf-8', newline='\n') as out:
                out.write(json.dumps(report, ensure_ascii=False, indent=2)[:-2])
                out.write(',\n  "skipped_files": [\n')
                first = True
                for (payload,) in db.execute('SELECT payload FROM skipped_files ORDER BY rowid'):
                    if not first:
                        out.write(',\n')
                    out.write('    ' + payload)
                    first = False
                out.write('\n  ]\n}\n')
            db.close()

            
            old_shards = OUTPUT / 'shards'
            backup = OUTPUT / '.shards_ancienne_version'
            if backup.exists():
                shutil.rmtree(backup)
            if old_shards.exists():
                os.replace(old_shards, backup)
            try:
                os.replace(stage / 'shards', old_shards)
            except BaseException:
                if backup.exists() and not old_shards.exists():
                    os.replace(backup, old_shards)
                raise
            for name in ('shards_manifest.json', 'file_inventory.jsonl',
                         'visual_manifest.jsonl', 'report.json'):
                os.replace(stage / name, OUTPUT / name)
            if backup.exists():
                shutil.rmtree(backup)
            for obsolete in ('pretrain_train.jsonl', 'pretrain_validation.jsonl',
                             'conversations_train.jsonl', 'conversations_validation.jsonl',
                             'corpus_tokenizer.txt'):
                (OUTPUT / obsolete).unlink(missing_ok=True)
        finally:
            try:
                db.close()
            except Exception:
                pass
    print(f'Data prepared: {pretrain_count:,} segments and {sft_count:,} conversations.')
    print(f'Files read: {processed:,}, skipped: {skipped_count:,}.')
    print(f'Shards : {sum(len(w.files) for w in writers.values())} files, directory: {OUTPUT / "shards"}')
    print(f'Manifest : {OUTPUT / "shards_manifest.json"}')
    
    try:
        from prepare_vision import prepare_vision
        visual = prepare_vision(SOURCES, OUTPUT, validation=args.validation,
                               shard_mb=args.shard_mb, root=ROOT)
        print('Vision prepared:', visual['annotated_images'], 'annotated example(s).')
    except ImportError as exc:
        print('Vision dataset not prepared (optional dependency or preparer missing):', exc)
    print('No model weights changed; images have not been trained.')


if __name__ == '__main__':
    main()
