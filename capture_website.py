#!/usr/bin/env python3
"""Capture responsive web screenshots and DOM sections with Playwright."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from vision_integrity import save_capture, sha256_file, utc_timestamp
from browser_support import (add_browser_argument, browser_info, launch_browser,
                             normalize_browser, normalize_browsers)

VIEWPORTS = {"mobile": (375, 812), "tablet": (768, 1024), "desktop": (1440, 900)}


def verify_url(url: str, allow_external: bool) -> None:
    p = urlparse(url)
    if p.scheme not in ('http', 'https') or not p.hostname:
        raise ValueError('A valid HTTP(S) URL is required.')
    host = p.hostname.lower()
    if not allow_external and host not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Only local URLs are allowed by default. --allow-external explicitly permits external URLs.')


def project_name_for_url(url: str) -> str:
    """Implement project name for url for the local model pipeline."""
    p = urlparse(url)
    host = (p.hostname or "site").replace(".", "-")
    path = (p.path or "/").strip("/").replace("/", "-")
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return slugify(f"{host}-{path}-{digest}")


def slugify(value: str) -> str:
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', value).strip('-').lower()
    if not slug or slug in ('.', '..'):
        raise ValueError('Invalid project name')
    return slug[:60]


def load_interactions(path: Path | None) -> list[dict]:
    if not path:
        return []
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict) or not isinstance(data.get('scenarios'), list):
        raise ValueError('Expected input JSON: {"scenarios": [...]}')
    valid_actions = {'click', 'fill', 'press'}
    for task in data['scenarios']:
        if not isinstance(task, dict) or task.get('action') not in valid_actions or not isinstance(task.get('selector'), str):
            raise ValueError('Invalid scenario: click/fill/press action and selector are required')
        if task['action'] in ('fill', 'press') and not isinstance(task.get('value'), str):
            raise ValueError('fill/press requires a value')
    return data['scenarios']


def run_scenarios(page, scenarios: list[dict]) -> list[dict]:
    results = []
    for item in scenarios:
        selector, action = item['selector'], item['action']
        result = {'action': action, 'selector': selector, 'ok': False}
        try:
            loc = page.locator(selector).first
            if action == 'click':
                loc.click(timeout=5000)
            elif action == 'fill':
                loc.fill(item['value'], timeout=5000)
            else:
                loc.press(item['value'], timeout=5000)
            expected = item.get('expected_visible')
            if expected and not page.locator(expected).first.is_visible(timeout=5000):
                raise AssertionError(f'Expected visible element is missing: {expected}')
            result['ok'] = True
        except Exception as e:
            result['error'] = str(e)[:500]
        results.append(result)
    return results


AUDIT_JS = r'''() => {
 const problematicImages = [...document.querySelectorAll('img')].filter(x => !x.hasAttribute('alt'));
 const headings = [...document.querySelectorAll('h1')].map(x => x.innerText.slice(0, 120));
 const overflow = document.documentElement.scrollWidth > window.innerWidth + 1;
 const unlabeledButtons = [...document.querySelectorAll('button')].filter(x =>
   !(x.getAttribute('aria-label') || x.innerText || '').trim()).length;
 return {
   title: document.title, url: location.href,
   h1: headings, images_without_alt: problematicImages.length,
   unlabeled_buttons: unlabeledButtons,
   horizontal_overflow: overflow,
   document_width: document.documentElement.scrollWidth,
   viewport_width: window.innerWidth
 };
}'''




SECTION_SELECTOR = (
    'header, main > section, main section, body > section, '
    'article, footer, nav, [role="banner"], [role="main"], '
    '[role="contentinfo"], [role="navigation"], form'
)


def create_annotation(image_file: Path, *, project: str, width: int,
                      element: dict | None = None, browser_metadata: dict | None = None) -> Path:
    """Implement create annotation for the local model pipeline."""
    sidecar = image_file.with_suffix('.vision.json')
    digest = sha256_file(image_file)
    if sidecar.exists():
        existing = json.loads(sidecar.read_text(encoding='utf-8'))
        if existing.get('image_sha256') != digest:
            raise ValueError('Existing annotation belongs to a different screenshot version: '+str(sidecar))
        return sidecar  
    data = {
        'project': project, 'phase': 'capture_pending_review',
        'image_sha256': digest, 'captured_at_utc': utc_timestamp(),
        'viewport_width': width,
        'question': 'Analyze this web interface and suggest visual and technical fixes.',
        'description': '', 'critique': '', 'improvements': '', 'correction': '',
        'code_files': [],
    }
    if browser_metadata is not None:
        data['browser_metadata'] = dict(browser_metadata)
    if element is not None:
        data['detected_element'] = element  
    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return sidecar


def capture_sections(page, dest: Path, *, project: str, viewport_label: str,
                     viewport_width: int, max_sections: int = 16,
                     browser_metadata: dict | None = None) -> list[dict]:
    """Implement capture sections for the local model pipeline."""
    if max_sections < 0 or max_sections > 100:
        raise ValueError('max_sections must be between 0 and 100')
    if max_sections == 0:
        return []
    locator = page.locator(SECTION_SELECTOR)
    results = []
    count = min(locator.count(), 400)  
    for i in range(count):
        if len(results) >= max_sections:
            break
        element = locator.nth(i)
        try:
            if not element.is_visible():
                continue
            bounds = element.bounding_box()
            if not bounds:
                continue
            width, height = bounds['width'], bounds['height']
            if width < 120 or height < 60 or height > 4500 or width * height > 13_000_000:
                continue
            meta = element.evaluate("""el => ({
                tag: el.tagName.toLowerCase(),
                role: el.getAttribute('role') || '',
                id: (el.id || '').slice(0, 80),
                classes: [...el.classList].slice(0, 5).join(' ').slice(0, 120),
                title: (el.querySelector('h1,h2,h3')?.textContent || el.getAttribute('aria-label') || '').trim().slice(0, 140)
            })""")
            safe_tag = slugify(meta.get('tag', 'section'))
            name = f'{viewport_label}-{viewport_width}-section-{len(results)+1:02d}-{safe_tag}'
            pixels = element.screenshot(animations='disabled', timeout=15000)
            screenshot, digest = save_capture(dest, name, pixels)
            create_annotation(screenshot, project=project, width=viewport_width, element=meta,
                              browser_metadata=browser_metadata)
            results.append({'capture': screenshot.name, 'image_sha256': digest, 'element': meta,
                            'width_px': int(width), 'height_px': int(height)})
        except Exception as exc:
            results.append({'error': str(exc)[:200], 'dom_position': i})
            
    return results


def capture(url: str, project: str | None, project_root: Path, *, allow_external: bool = False,
            scenarios: list[dict] | None = None, timeout: int = 30_000,
            max_sections: int = 16, browser: str = 'chromium') -> dict:
    selected = normalize_browser(browser)
    if max_sections < 0 or max_sections > 100:
        raise ValueError('max_sections must be between 0 and 100')
    verify_url(url, allow_external)
    project = slugify(project) if project else project_name_for_url(url)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError('Install Playwright: python -m pip install playwright; then install the selected browsers as described in VISION_AND_BROWSER.md') from e
    dest = project_root / 'training_data' / 'vision' / project / selected
    dest.mkdir(parents=True, exist_ok=True)
    audit_dir = project_root / 'outputs' / 'audits_browser'
    audit_dir.mkdir(parents=True, exist_ok=True)
    metadata = browser_info(selected)
    report = {'project': project, 'url': url, 'captured_at_utc': datetime.now(timezone.utc).isoformat(),
              **metadata, 'results': [], 'warning': 'Technical diagnostics, not aesthetic judgments.'}
    with sync_playwright() as p:
        active_browser = launch_browser(p, selected)
        metadata['browser_version'] = active_browser.version
        report.update(metadata)
        try:
            for label, (width, height) in VIEWPORTS.items():
                context = active_browser.new_context(viewport={'width': width, 'height': height}, device_scale_factor=1,
                                              reduced_motion='reduce')
                page = context.new_page()
                errors, failed = [], []
                page.on('pageerror', lambda error: errors.append(str(error)[:500]))
                page.on('console', lambda msg: errors.append('Console error: '+msg.text[:400]) if msg.type == 'error' else None)
                page.on('requestfailed', lambda req: failed.append(req.url[:300]))
                row = {'viewport': label, 'width': width, 'height': height, **metadata}
                try:
                    response = page.goto(url, wait_until='domcontentloaded', timeout=timeout)
                    page.wait_for_timeout(500)
                    row['http_status'] = response.status if response else None
                    row['audit'] = page.evaluate(AUDIT_JS)
                    row['interactions'] = run_scenarios(page, scenarios or [])
                    pixels = page.screenshot(full_page=True, animations='disabled')
                    image_file, image_sha = save_capture(dest, f'{label}-{width}', pixels)
                    row['capture'] = image_file.relative_to(project_root).as_posix()
                    row['image_sha256'] = image_sha
                    create_annotation(image_file, project=project, width=width, browser_metadata=metadata)
                    row['sections'] = capture_sections(page, dest, project=project,
                                                       viewport_label=label, viewport_width=width,
                                                       max_sections=max_sections, browser_metadata=metadata)
                except Exception as exc:
                    row['error'] = str(exc)[:500]
                row['javascript_errors'] = errors
                row['failed_requests'] = failed
                report['results'].append(row)
                context.close()
        finally:
            active_browser.close()
    (audit_dir / f'{project}-{selected}.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    return report


def capture_browsers(url: str, project: str | None, project_root: Path, *, browsers=None, **options) -> dict:
    selected = normalize_browsers(browsers)
    reports = [capture(url, project, project_root, browser=name, **options) for name in selected]
    return {'project': reports[0]['project'], 'url': url, 'browsers': list(selected),
            'browser_runs': [{key: value for key, value in report.items() if key != 'results'}
                             for report in reports],
            'results': [row for report in reports for row in report['results']]}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Playwright: capture mobile/tablet/desktop viewports and run technical checks.')
    parser.add_argument('url', help='Example: http://localhost:3000')
    add_browser_argument(parser)
    parser.add_argument('--project', help='Optional project name; defaults to a URL-derived name')
    parser.add_argument('--max-sections', type=int, default=16, help='Maximum captured HTML sections per viewport (default 16)')
    parser.add_argument('--no-sections', action='store_true', help='Disable automatic HTML section captures')
    parser.add_argument('--allow-external', action='store_true', help='Explicitly allow external URLs you are authorized to inspect')
    parser.add_argument('--interactions', type=Path, help='JSON with explicit click/fill/press interaction scenarios')
    args = parser.parse_args(argv)
    if args.max_sections < 0 or args.max_sections > 100:
        parser.error('--max-sections must be between 0 and 100')
    report = capture_browsers(args.url, args.project, Path(__file__).resolve().parent, browsers=args.browser,
                     allow_external=args.allow_external, scenarios=load_interactions(args.interactions),
                     max_sections=0 if args.no_sections else args.max_sections)
    for run in report['browser_runs']:
        print(f"{run['browser_label']} | version: {run['browser_version']}")
    for item in report['results']:
        print(f"{item['browser']}/{item['viewport']}: {item.get('capture', 'FAILED')} | sections: {len([r for r in item.get('sections',[]) if r.get('capture')])} | JS: {len(item['javascript_errors'])} | interactions: {len(item.get('interactions', []))}")
    print('Add reviewed descriptions and critiques to the .vision.json files before training.')


if __name__ == '__main__':
    main()
