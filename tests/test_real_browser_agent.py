"""Real browser integration test using Playwright on Windows, Linux and macOS.

Install browsers with: python -m playwright install chromium firefox webkit.
Set CHAMP_LLM_TEST_BROWSERS=chromium,chrome,firefox,safari to also exercise installed Chrome.
Missing browser dependencies or blocked localhost access must fail this test.
"""
import json
import os
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import coding_agent as agent
from browser_support import normalize_browsers


class SilentHandler(SimpleHTTPRequestHandler):
    def log_message(self, *a, **k):
        pass


class BrowserLoopActualTests(unittest.TestCase):
    def test_real_capture_modify_and_recheck(self):
        browsers = normalize_browsers(os.environ.get('CHAMP_LLM_TEST_BROWSERS', 'chromium,firefox,webkit').split(','))
        with tempfile.TemporaryDirectory(prefix='agent-browser-') as td:
            root = Path(td)
            project = root/'site'
            project.mkdir()
            initial = ('<!doctype html><html><head><meta charset="utf-8">'
                       '<link rel="icon" href="data:,"><title>Test</title></head>'
                       '<body><main style="width:3500px"><section style="min-height:150px">'
                       '<h1>Débordement</h1>'
                       '<form onsubmit="event.preventDefault(); document.getElementById(&quot;result&quot;).hidden=false">'
                       '<input id="name" aria-label="Name"><button id="show" type="submit">Show</button>'
                       '<p id="result" hidden>Ready</p></form></section></main></body></html>')
            corrected = initial.replace('width:3500px', 'max-width:100%;').replace('Débordement', 'Corrigé')
            scenarios = root / 'interactions.json'
            scenarios.write_text(json.dumps({'scenarios': [
                {'action':'fill','selector':'#name','value':'Browser test'},
                {'action':'press','selector':'#name','value':'Enter','expected_visible':'#result'},
                {'action':'click','selector':'#show','expected_visible':'#result'},
            ]}), encoding='utf-8')
            (project/'index.html').write_text(initial, encoding='utf-8')
            server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SilentHandler, directory=str(project)))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            answers = iter([
                json.dumps({'edits':[{'path':'index.html','content':corrected}],
                            'comment':'Corriger le débordement'}),
                json.dumps({'edits':[], 'comment':'Terminé'}),
            ])
            try:
                # Every selected browser is required; missing browsers are failures, not skips.
                with patch.object(agent, 'ROOT', root), \
                     patch.object(agent, 'ollama_chat', side_effect=lambda *a, **k:next(answers)):
                    result = agent.run_agent(project, f'http://127.0.0.1:{server.server_port}/index.html',
                                             'Corriger le responsive',code_model='mock',visual_model=None,
                                             test_command=None,interactions=scenarios,apply=True,iterations=2, browsers=browsers)
                self.assertEqual(result['browsers'], list(browsers))
                self.assertEqual(len(result['cycles']),2)
                first, second = result['cycles']
                self.assertCountEqual(first['issues_before'], [
                    f'{browser}/{viewport}: horizontal overflow'
                    for browser in browsers for viewport in ('mobile', 'tablet', 'desktop')
                ])
                self.assertEqual(first['files'], ['index.html'])
                self.assertEqual(first['issues_after'], [])
                self.assertEqual(second['issues_before'], [])
                self.assertEqual(second['issues_after'], [])
                self.assertEqual(second['files'], [])
                self.assertEqual((project/'index.html').read_text(encoding='utf-8'), corrected)

                run_root = Path(result['report_path']).parent
                audit_files = list((run_root/'outputs'/'audits_browser').glob('*.json'))
                self.assertEqual(len(audit_files), len(browsers))
                seen = set()
                captures = set()
                for audit_file in audit_files:
                    audit = json.loads(audit_file.read_text(encoding='utf-8'))
                    browser = audit['browser']
                    seen.add(browser)
                    self.assertTrue(audit['browser_version'])
                    print(f"Browser under test: {browser} | {audit['browser_version']} | engine={audit['engine']}", flush=True)
                    self.assertEqual(audit['engine'], 'chromium' if browser == 'chrome' else browser)
                    self.assertEqual(audit['channel'], 'chrome' if browser == 'chrome' else None)
                    if browser == 'webkit':
                        self.assertIn('not the Safari application', audit['browser_label'])
                    self.assertCountEqual([row['viewport'] for row in audit['results']],
                                          ['mobile', 'tablet', 'desktop'])
                    for row in audit['results']:
                        with self.subTest(browser=browser, viewport=row['viewport']):
                            self.assertNotIn('error', row)
                            self.assertEqual(row['browser'], browser)
                            self.assertEqual(row['http_status'], 200)
                            self.assertEqual(row['failed_requests'], [])
                            self.assertEqual(row['javascript_errors'], [])
                            self.assertEqual(row['audit']['h1'], ['Corrigé'])
                            self.assertFalse(row['audit']['horizontal_overflow'])
                            self.assertEqual(len(row['interactions']), 3)
                            self.assertTrue(all(action['ok'] for action in row['interactions']))
                            screenshot = run_root/row['capture']
                            self.assertEqual(screenshot.parent.name, browser)
                            self.assertGreater(screenshot.stat().st_size, 0)
                            self.assertNotIn(screenshot, captures)
                            captures.add(screenshot)
                            annotation = json.loads(screenshot.with_suffix('.vision.json').read_text(encoding='utf-8'))
                            self.assertEqual(annotation['browser_metadata']['browser'], browser)
                            self.assertEqual(annotation['browser_metadata']['browser_version'], audit['browser_version'])
                            self.assertEqual(len(list(screenshot.parent.glob(
                                f"{row['viewport']}-{row['width']}-" + "?" * 16 + ".png"))), 2)
                            self.assertTrue(row['sections'])
                            self.assertTrue(all('capture' in section for section in row['sections']))
                self.assertEqual(seen, set(browsers))
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)
