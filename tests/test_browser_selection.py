"""Browser routing, failures and per-browser agent feedback."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import browser_support as support
import capture_website as capture
import coding_agent as agent


class BrowserSelectionTests(unittest.TestCase):
    def engines(self):
        return SimpleNamespace(chromium=Mock(), firefox=Mock(), webkit=Mock())

    def test_aliases_are_explicit_deduplicated_and_invalid_choices_rejected(self):
        self.assertEqual(support.normalize_browsers(), ('chromium',))
        self.assertEqual(support.normalize_browsers(['chrome','safari','firefox','webkit']),
                         ('chrome','webkit','firefox'))
        self.assertEqual(support.browser_info('safari')['engine'], 'webkit')
        self.assertIn('not the Safari application', support.browser_info('safari')['browser_label'])
        for value in ('edge', '../firefox', [], ['chromium', 'bad']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                support.normalize_browsers(value)

    def test_chrome_uses_the_official_channel_and_ignores_chromium_override(self):
        engines = self.engines()
        with patch.dict(os.environ, {'CHAMP_LLM_CHROMIUM':'custom-chromium'}, clear=True):
            support.launch_browser(engines, 'chrome')
        engines.chromium.launch.assert_called_once_with(headless=True, channel='chrome')
        engines.firefox.launch.assert_not_called()
        engines.webkit.launch.assert_not_called()

    def test_firefox_and_safari_use_their_own_engines_without_chromium_flags(self):
        engines = self.engines()
        with patch.dict(os.environ, {'CHAMP_LLM_CHROMIUM':'custom-chromium'}, clear=True):
            support.launch_browser(engines, 'firefox')
            support.launch_browser(engines, 'safari')
        engines.firefox.launch.assert_called_once_with(headless=True)
        engines.webkit.launch.assert_called_once_with(headless=True)
        engines.chromium.launch.assert_not_called()

    def test_chromium_override_priority_and_legacy_fallback(self):
        for variables, expected in (
            ({'CHAMP_LLM_CHROMIUM':'new','CHAMP_GAUCHE_LLM_CHROMIUM':'old'}, 'new'),
            ({'CHAMP_GAUCHE_LLM_CHROMIUM':'old','MONCERVEAU_CHROMIUM':'legacy'}, 'old'),
            ({'MONCERVEAU_CHROMIUM':'legacy'}, 'legacy'),
        ):
            with self.subTest(variables=variables), patch.dict(os.environ, variables, clear=True):
                engines = self.engines()
                support.launch_browser(engines, 'chromium')
                engines.chromium.launch.assert_called_once_with(headless=True, executable_path=expected, args=['--no-sandbox'])

    def test_missing_engine_fails_with_install_help_and_never_falls_back(self):
        for browser, engine in (('chrome','chromium'), ('firefox','firefox'), ('safari','webkit')):
            with self.subTest(browser=browser):
                engines = self.engines()
                getattr(engines, engine).launch.side_effect = RuntimeError('missing executable')
                with self.assertRaisesRegex(RuntimeError, 'No fallback was used'):
                    support.launch_browser(engines, browser)
                for other in ('chromium','firefox','webkit'):
                    if other != engine:
                        getattr(engines, other).launch.assert_not_called()

    def test_both_clis_forward_repeated_browser_choices(self):
        with patch.object(capture, 'capture_browsers', return_value={'browser_runs':[], 'results':[]}) as calls, contextlib.redirect_stdout(io.StringIO()):
            capture.main(['http://localhost:3000', '--browser','chrome','--browser','safari'])
        self.assertEqual(calls.call_args.kwargs['browsers'], ['chrome','safari'])
        with patch.object(agent, 'run_agent', return_value={'cycles':[], 'report_path':'report.json'}) as calls, contextlib.redirect_stdout(io.StringIO()):
            agent.main(['--project','site','--url','http://localhost:3000','--goal','Fix layout',
                        '--backend','ollama','--code-model','test', '--browser','firefox','--browser','safari'])
        self.assertEqual(calls.call_args.kwargs['browsers'], ['firefox','safari'])

    def test_failure_stops_the_agent_before_generating_or_applying_edits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            site = root / 'site'
            site.mkdir()
            page = site / 'index.html'
            page.write_text('<h1>Keep me</h1>', encoding='utf-8')
            with patch.object(agent, 'ROOT', root), patch.object(agent, 'capture_browsers', side_effect=RuntimeError('Cannot launch Firefox')), patch.object(agent, 'ollama_chat') as model:
                with self.assertRaisesRegex(RuntimeError, 'Cannot launch Firefox'):
                    agent.run_agent(site, 'http://localhost:3000','Fix layout', code_model='test',
                                    visual_model=None, test_command=None, interactions=None,
                                    apply=True, iterations=1, browsers=['firefox'])
            model.assert_not_called()
            self.assertEqual(page.read_text(), '<h1>Keep me</h1>')
            report = json.loads(next((root / 'outputs/agent').glob('*/report.json')).read_text())
            self.assertIn('Cannot launch Firefox', report['error'])

    def test_issues_keep_the_browser_identity(self):
        report = {'results': [{'browser':name,'viewport':'mobile','audit':{'horizontal_overflow':True}}
                              for name in ('chrome','firefox','webkit')]}
        self.assertEqual(agent.problems(report), [f'{name}/mobile: horizontal overflow' for name in ('chrome','firefox','webkit')])

    def test_vision_reviews_each_browser_and_keeps_mobile_and_desktop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for browser in ('chrome','firefox','webkit'):
                folder = root / browser
                folder.mkdir()
                for viewport in ('mobile','tablet','desktop'):
                    path = folder / f'{viewport}.png'
                    path.write_bytes(b'image')
                    rows.append({'browser':browser,'viewport':viewport,'capture':path.relative_to(root).as_posix()})
            backend = Mock()
            backend.critique_images.return_value = 'Review ' * 2000
            result = agent.visual_feedback(root, {'results':rows}, None, local_backend=backend)
            self.assertEqual(backend.critique_images.call_count, 3)
            self.assertLessEqual(len(result), 9000)
            for browser, call in zip(('chrome','firefox','webkit'), backend.critique_images.call_args_list):
                self.assertEqual(call.args[0], [(root / browser / f'{viewport}.png').resolve() for viewport in ('mobile','desktop')])
                self.assertIn(f'[{browser}]', result)


if __name__ == '__main__':
    unittest.main()
