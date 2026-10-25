#!/usr/bin/env python3
"""Iteratively inspect, edit, and test websites with a local or Ollama coding backend."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from capture_website import capture_browsers, load_interactions, verify_url
from browser_support import add_browser_argument, normalize_browsers

ROOT = Path(__file__).resolve().parent
ALLOWED_SUFFIXES = frozenset({
    '.jsx','.tsx','.js','.mjs','.cjs','.ts','.css','.scss','.sass','.html',
    '.php','.vue','.svelte','.astro','.json','.md','.txt','.py','.sql','.yml','.yaml',
})
BLOCKED_PARTS = frozenset({
    '.git','.venv','node_modules','vendor','dist','build','.next','.nuxt',
    '.cache','.env','.github','.vscode','coverage','__pycache__','secrets',
    'outputs','training_runs','model','demo_model','agent_runs',
})
MAX_BYTES_PER_FILE = 200_000
MAX_EDITS = 12


def allowed_file(project: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or len(relative)>500 or '\\' in relative:
        raise ValueError('Invalid file path')
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or any(part.startswith('.') or part in BLOCKED_PARTS for part in rel.parts):
        raise ValueError(f'Disallowed path: {relative}')
    if rel.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f'Disallowed extension : {relative}')
    path = (project.resolve() / rel).resolve()
    if not path.is_relative_to(project.resolve()):
        raise ValueError('Path escapes the project directory (possible symlink)')
    if path.exists() and (not path.is_file() or path.stat().st_size > MAX_BYTES_PER_FILE):
        raise ValueError('File too large or not regular')
    return path


def parse_edits(output: str) -> tuple[list[dict], str]:
    text = output.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get('edits'), list):
        raise ValueError('Expected JSON response: {"edits": [{"path":..., "content":...}], "comment": ...}')
    edits = data['edits']
    if len(edits) > MAX_EDITS:
        raise ValueError(f'Too many modified files : maximum {MAX_EDITS} par cycle')
    seen = set()
    for edit in edits:
        if not isinstance(edit, dict) or not isinstance(edit.get('path'), str) or not isinstance(edit.get('content'), str):
            raise ValueError('Each edit must include path and content (the full file text)')
        if len(edit['content'].encode('utf-8')) > MAX_BYTES_PER_FILE:
            raise ValueError('Model proposed an oversized file')
        if edit['path'] in seen:
            raise ValueError('File modified twice in one cycle')
        seen.add(edit['path'])
    return edits, str(data.get('comment', ''))[:3000]


def apply_edits(project: Path, edits: list[dict], backups: Path, *, apply: bool) -> list[str]:
    resolved = [(allowed_file(project, e['path']), e['content']) for e in edits]
    changed = []
    for path, content in resolved:
        relative = path.relative_to(project.resolve())
        if path.exists() and path.read_text(encoding='utf-8') == content:
            continue
        changed.append(relative.as_posix())
        if not apply:
            continue
        if path.exists():
            backup = backups / relative
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=path.parent, prefix='.agent-', suffix='.tmp')
        try:
            with os.fdopen(fd, 'w', encoding='utf-8', newline='\n') as out:
                out.write(content)
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return changed


def snapshot(project: Path, *, max_chars: int = 42000, max_files: int = 200) -> str:
    """Implement snapshot for the local model pipeline."""
    entries = []
    used = 0
    count = 0
    for parent, dirs, files in os.walk(project, topdown=True, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in BLOCKED_PARTS and not d.startswith('.')
                         and not (Path(parent)/d).is_symlink())
        for filename in sorted(files):
            if count >= max_files or used >= max_chars:
                return '\n'.join(entries) or '(No readable source files)'
            path = Path(parent)/filename
            if path.is_symlink():
                continue
            rel = path.relative_to(project)
            try:
                allowed_file(project, rel.as_posix())
                if path.stat().st_size > 120_000:
                    continue
                content = path.read_text(encoding='utf-8')
            except (ValueError, OSError, UnicodeError):
                continue
            room = max_chars-used
            if room < 200:
                break
            content = content[:max(0, room-len(str(rel))-64)]
            entries.append(f'=== {rel.as_posix()} ===\n{content}\n')
            used += len(entries[-1])
            count += 1
    return '\n'.join(entries) or '(No readable source files)'


def run_tests(project: Path, command: str | None, *, timeout: int = 180) -> dict:
    if not command:
        return {'status': 'non_configure', 'stdout': '', 'stderr': ''}
    if timeout < 1:
        raise ValueError('Invalid timeout')
    args = shlex.split(command, posix=os.name != 'nt')
    if not args:
        raise ValueError('Empty test command')
    
    try:
        completed = subprocess.run(args, cwd=project, capture_output=True, text=True,
                                   timeout=timeout, shell=False, errors='replace')
        return {'status': 'reussi' if completed.returncode == 0 else 'echec',
                'exit_code': completed.returncode, 'stdout': completed.stdout[-6000:],
                'stderr': completed.stderr[-6000:]}
    except subprocess.TimeoutExpired:
        return {'status': 'timeout', 'stdout': '', 'stderr': 'Timeout exceeded'}
    except OSError as exc:
        return {'status': 'echec', 'stdout': '', 'stderr': str(exc)[:2000]}


def ollama_chat(model: str, user_text: str, *, image_paths: list[Path] | None = None,
                endpoint: str = 'http://127.0.0.1:11434', timeout: int = 180) -> str:
    target = urlparse(endpoint)
    if target.scheme != 'http' or target.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Only local Ollama is permitted to avoid disclosing project code')
    message = {'role':'user', 'content':user_text}
    if image_paths:
        message['images'] = [base64.b64encode(path.read_bytes()).decode('ascii') for path in image_paths]
    data = json.dumps({'model':model, 'stream': False, 'messages':[message],
                       'options': {'temperature': 0.1}}, ensure_ascii=False).encode('utf-8')
    request = Request(endpoint.rstrip('/') + '/api/chat', data=data, headers={'Content-Type':'application/json'})
    with urlopen(request, timeout=timeout) as response:
        content = json.load(response)
    return content['message']['content']


def inspect_site(url: str, run_root: Path, *, scenarios=None, browsers=None) -> dict:
    
    verify_url(url, allow_external=False)
    return capture_browsers(url, project='agent-'+run_root.name, project_root=run_root, browsers=browsers,
                   allow_external=False, scenarios=scenarios, max_sections=5)


def problems(report: dict) -> list[str]:
    observations = []
    for item in report.get('results', []):
        location = (str(item['browser']) + '/' if item.get('browser') else '') + str(item.get('viewport'))
        if 'error' in item:
            observations.append(f"{location}: failed to load: {item['error']}")
        if item.get('http_status') and item['http_status'] >= 400:
            observations.append(f"{location}: HTTP {item['http_status']}")
        for err in item.get('javascript_errors', [])[:5]:
            observations.append(f"{location}: JS {err}")
        audit = item.get('audit') or {}
        if audit.get('horizontal_overflow'):
            observations.append(f"{location}: horizontal overflow")
        if audit.get('images_without_alt'):
            observations.append(f"{location}: {audit['images_without_alt']} image(s) without alt text")
        if audit.get('unlabeled_buttons'):
            observations.append(f"{location}: unlabeled buttons: {audit['unlabeled_buttons']}")
        for action in item.get('interactions', []):
            if not action.get('ok'):
                observations.append(f"{location}: interaction failed: {action.get('error')}")
    return observations[:40]


def visual_feedback(run_root: Path, report: dict, model: str | None, *, local_backend=None) -> str:
    if not model and local_backend is None:
        return 'Visual review unavailable: specify --vision-model or --vision-checkpoint.'
    grouped = {}
    rows = sorted(report.get('results', []), key=lambda row: {'mobile': 0, 'desktop': 1, 'tablet': 2}.get(row.get('viewport'), 3))
    for item in rows:
        rel = item.get('capture')
        group = item.get('browser', 'browser')
        selected = grouped.setdefault(group, [])
        if rel and len(selected) < 2:
            path = (run_root / rel).resolve()
            if path.is_relative_to(run_root.resolve()) and path.is_file() and path.stat().st_size < 8_000_000:
                selected.append(path)
    grouped = {name: paths for name, paths in grouped.items() if paths}
    if not grouped:
        return 'No screenshots available for vision.'
    feedback = []
    budget = max(1, 8500 // len(grouped))
    for name, paths in grouped.items():
        if local_backend is not None:
            critique = local_backend.critique_images(paths)
        else:
            prompt = (f'Review these {name} screenshots for typography, alignment, overflow, '
                      'accessibility and responsiveness. Suggest precise CSS fixes. '
                      'Your review is not yet verified by a human.')
            critique = ollama_chat(model, prompt, image_paths=paths)
        feedback.append(f'[{name}]\n{critique[:budget]}')
    return '\n\n'.join(feedback)[:9000]


def run_agent(project: Path, url: str, goal: str, *, code_model: str, visual_model: str | None,
              test_command: str | None, interactions: Path | None, apply: bool,
              iterations: int, endpoint: str = 'http://127.0.0.1:11434',
              local_backend=None, max_source_chars: int = 42000, browsers=None) -> dict:
    browsers = normalize_browsers(browsers)
    project = project.expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f'Project not found: {project}')
    verify_url(url, allow_external=False)
    if not goal.strip():
        raise ValueError('Goal must not be empty')
    if not (1 <= iterations <= 12):
        raise ValueError('Use between 1 and 12 iterations')
    if not (500 <= max_source_chars <= 100000):
        raise ValueError('max_source_chars must be between 500 and 100000')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    run_root = ROOT / 'outputs' / 'agent' / run_id
    run_root.mkdir(parents=True)
    scenarios = load_interactions(interactions)
    report = {'goal':goal, 'project':str(project),
              'backend': 'champ-llm' if local_backend is not None else 'ollama',
              'code_model':code_model, 'vision_model':visual_model,
              'apply':apply, 'browsers':list(browsers), 'cycles':[]}
    feedback = ''
    try:
        for cycle in range(1, iterations+1):
            before = inspect_site(url, run_root, scenarios=scenarios, browsers=browsers)
            tests_before = run_tests(project, test_command)
            critique = visual_feedback(run_root, before, visual_model, local_backend=local_backend)
            prompt = (
                'You are a web development coding agent. Return ONLY valid JSON, no Markdown. '
                'Format: {"edits":[{"path":"src/App.jsx","content":"COMPLETE FILE CONTENT"}],'
                '"comment":"explanation"}. Edit at most 12 files, each at most 200000 bytes. '
                'Never write secrets, .env files, or shell commands. '
                'If no fix is needed return {"edits":[],"comment":"done"}. '
                'Use English identifiers in source code, but reply in the language of the user goal.\n\n'
                f'GOAL: {goal}\n'
                f'BROWSERS: {json.dumps(before.get("browser_runs", list(browsers)), ensure_ascii=False)}\n'
                f'BROWSER ISSUES: {json.dumps(problems(before), ensure_ascii=False)}\n'
                f'TEST RESULTS: {json.dumps(tests_before, ensure_ascii=False)[:9000]}\n'
                f'VISUAL REVIEW: {critique}\n'
                f'PREVIOUS ITERATION FEEDBACK: {feedback}\n'
                f'PROJECT FILES (truncated if oversized):\n{snapshot(project, max_chars=max_source_chars)}'
            )
            response = (local_backend.generate_code(prompt) if local_backend is not None
                        else ollama_chat(code_model, prompt, endpoint=endpoint))
            edits, comment = parse_edits(response)
            files = apply_edits(project, edits, run_root/'backups'/f'cycle-{cycle}', apply=apply)
            after = inspect_site(url, run_root, scenarios=scenarios, browsers=browsers) if apply and files else before
            tests_after = run_tests(project, test_command) if apply and files else tests_before
            item = {'cycle':cycle, 'files':files, 'model_comment':comment,
                    'tests_before':tests_before, 'tests_after':tests_after,
                    'issues_before':problems(before), 'issues_after':problems(after),
                    'browser_runs':before.get('browser_runs', []),
                    'unverified_visual_feedback':critique}
            report['cycles'].append(item)
            (run_root/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            feedback = json.dumps({'issues_after':problems(after), 'tests_after':tests_after}, ensure_ascii=False)[:9000]
            if not files or not apply:
                break
    except Exception as exc:
        report['error'] = str(exc)
        raise
    finally:
        report['report_path'] = str(run_root/'report.json')
        (run_root/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description='Local coding agent: browser inspection, critique, testing, and correction.')
    add_browser_argument(p)
    p.add_argument('--project', required=True, type=Path, help='Directory of a project that you own or control')
    p.add_argument('--url', required=True, help='http://localhost:3000 (start the site first)')
    p.add_argument('--goal', required=True, help='Creation or debugging goal')
    p.add_argument('--backend', choices=['local', 'ollama'], default='local',
                   help='local = load champ-llm weights directly (without Ollama)')
    p.add_argument('--local-model', type=Path, default=ROOT/'model',
                   help='Directory with config.json, tokenizer and trained .safetensors weights')
    p.add_argument('--vision-checkpoint', type=Path,
                   help='Vision run with latest.json, or a checkpoint containing vision.safetensors')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--max-generation', type=int, default=512)
    p.add_argument('--max-vision-generation', type=int, default=180)
    p.add_argument('--max-source-chars', type=int,
                   help='Maximum characters of code sent to model (default 6000 local, 42000 Ollama)')
    p.add_argument('--code-model', help='Ollama model name (only with --backend ollama)')
    p.add_argument('--vision-model', help='Ollama vision model (only with --backend ollama)')
    p.add_argument('--tests', help='Your project test command, e.g. npm run build')
    p.add_argument('--interactions', type=Path, help='Optional Playwright click/fill/press scenarios')
    p.add_argument('--cycles', type=int, default=3)
    p.add_argument('--apply', action='store_true', help='ALLOW the AI to write source files')
    args = p.parse_args(argv)
    local = None
    if args.backend == 'local':
        if args.code_model or args.vision_model:
            p.error('--code-model / --vision-model require --backend ollama')
        from local_backend import ChampBackend
        local = ChampBackend(args.local_model, vision=args.vision_checkpoint,
                                  device=args.device, max_generation=args.max_generation,
                                  max_vision_generation=args.max_vision_generation)
    else:
        if not args.code_model:
            p.error('--code-model is required with --backend ollama')
        if args.vision_checkpoint:
            p.error('--vision-checkpoint requires --backend local')
    max_sources = (6000 if local is not None else 42000) if args.max_source_chars is None else args.max_source_chars
    result = run_agent(args.project, args.url, args.goal,
                       code_model=str(args.local_model) if local is not None else args.code_model,
                       visual_model=str(args.vision_checkpoint) if local is not None and args.vision_checkpoint else args.vision_model,
                       test_command=args.tests, interactions=args.interactions, apply=args.apply,
                       iterations=args.cycles, local_backend=local, max_source_chars=max_sources, browsers=args.browser)
    print('Cycles :', len(result['cycles']))
    print('Rapport :', result['report_path'])
    if not args.apply:
        print('Preview mode: no edits applied. Use --apply only after review.')


if __name__ == '__main__':
    main()
