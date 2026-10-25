import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import coding_agent as agent


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)/'site'
        self.project.mkdir()
        (self.project/'index.html').write_text('<h1>Avant</h1>', encoding='utf-8')

    def test_path_protection(self):
        for name in ('../secret.py','.env','node_modules/a.js','foo.exe','/etc/passwd','a\\b.js'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                agent.allowed_file(self.project,name)
        self.assertEqual(agent.allowed_file(self.project,'src/App.jsx'),(self.project/'src'/'App.jsx').resolve())

    def test_json_validation(self):
        edits, _ = agent.parse_edits('```json\n{"edits":[{"path":"index.html","content":"<h1>OK</h1>"}]}\n```')
        self.assertEqual(edits[0]['path'],'index.html')
        with self.assertRaises(ValueError):
            agent.parse_edits('{"edits":[{"path":"x.php","content":1}]}')

    def test_apply_and_backup(self):
        backup = Path(self.tmp.name)/'sauvegarde'
        edits = [{'path':'index.html','content':'<h1>Après</h1>'}]
        changed = agent.apply_edits(self.project,edits,backup,apply=False)
        self.assertEqual(changed,['index.html'])
        self.assertEqual((self.project/'index.html').read_text(encoding='utf-8'),'<h1>Avant</h1>')
        agent.apply_edits(self.project,edits,backup,apply=True)
        self.assertEqual((self.project/'index.html').read_text(encoding='utf-8'),'<h1>Après</h1>')
        self.assertEqual((backup/'index.html').read_text(encoding='utf-8'),'<h1>Avant</h1>')

    def test_two_iteration_mock_loop(self):
        reports = [
            {'results':[{'viewport':'mobile','javascript_errors':['error']}]},
            {'results':[{'viewport':'mobile','javascript_errors':[]}]}
        ]
        answers = iter([
            json.dumps({'edits':[{'path':'index.html','content':'<h1>Après</h1>'}], 'comment':'Corrigé'}),
            json.dumps({'edits':[], 'comment':'OK'})
        ])
        root = Path(self.tmp.name)/'run_root'
        with patch.object(agent,'ROOT',root), \
             patch.object(agent,'inspect_site',side_effect=[reports[0], reports[1], reports[1]]), \
             patch.object(agent,'ollama_chat',side_effect=lambda *a,**kw: next(answers)):
            result = agent.run_agent(self.project,'http://localhost:3000','Corriger',
                                     code_model='test',visual_model=None,test_command=None,
                                     interactions=None,apply=True,iterations=3)
        self.assertEqual(len(result['cycles']),2)
        self.assertEqual((self.project/'index.html').read_text(encoding='utf-8'),'<h1>Après</h1>')
        self.assertTrue(Path(result['report_path']).is_file())


if __name__ == '__main__':
    unittest.main()
