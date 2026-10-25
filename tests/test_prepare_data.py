"""Tools for the champ-llm local training pipeline."""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / 'prepare_data.py'
spec = importlib.util.spec_from_file_location('preparer_donnees_test', SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.original = (mod.ROOT, mod.SOURCES, mod.SINGLE, mod.OUTPUT)
        mod.ROOT = self.base
        mod.SOURCES = self.base / 'training_data'
        mod.SINGLE = self.base / 'training_data.txt'
        mod.OUTPUT = self.base / 'outputs'
        mod.SOURCES.mkdir()

    def tearDown(self):
        mod.ROOT, mod.SOURCES, mod.SINGLE, mod.OUTPUT = self.original
        self.temp.cleanup()

    def add(self, rel, content):
        path = mod.SOURCES / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode('utf-8'))
        return path

    def manifest(self):
        return json.loads((mod.OUTPUT / 'shards_manifest.json').read_text(encoding='utf-8'))

    def rows(self, split):
        result = []
        for row in self.manifest()['splits'][split]['files']:
            shard = mod.OUTPUT / row['path']
            self.assertTrue(shard.is_file(), f'shard absent: {shard}')
            if split == 'tokenizer':
                result.append(shard.read_text(encoding='utf-8'))
            else:
                result += [json.loads(x) for x in shard.read_text(encoding='utf-8').splitlines() if x]
        return result

    def pretrain(self):
        return self.rows('pretrain_train') + self.rows('pretrain_validation')

    def test_nested_extensions_and_metadata(self):
        self.add('project/ui/react/composants/boutons/Button.jsx', 'export default function Bouton(){return <button>OK</button>}')
        self.add('project/php/admin/index.php', "<?php echo 'salut';")
        self.add('project/styles/anim.monlanguage', '.couleur { color: red; }')
        mod.SINGLE.write_text('Bonjour !', encoding='utf-8')
        mod.main([])
        docs = self.pretrain()
        rendered = '\n'.join(d['text'] for d in docs)
        self.assertIn('Path: training_data/project/ui/react/composants/boutons/Button.jsx', rendered)
        self.assertIn('Language: React JSX', rendered)
        self.assertIn('Language: PHP', rendered)
        self.assertIn('anim.monlanguage', rendered)
        self.assertIn('Bonjour !', rendered)
        self.assertEqual(4, len(docs))
        self.assertTrue(self.rows('tokenizer'))

    def test_jsonl_chat_and_visual_manifest(self):
        messages = {'messages': [
            {'role': 'user', 'content': 'Bonjour'},
            {'role': 'assistant', 'content': 'Salut'},
        ]}
        self.add('conversations/questions.jsonl', json.dumps(messages, ensure_ascii=False))
        self.add('captures/site.png', b'\x89PNG\r\n\x1a\n')
        self.add('captures/site.caption.txt', 'Site sobre, lisible et contrasté')
        mod.main([])
        self.assertEqual(1, len(self.rows('conversations_train')))
        self.assertTrue(any('<|user|>' in x['text'] for x in self.pretrain()))
        visual = [json.loads(x) for x in (mod.OUTPUT / 'visual_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(1, len(visual))
        self.assertTrue(visual[0]['annotation_complete'])

    def test_sft_learning_flags_preserved(self):
        messages = {'messages': [
            {'role':'user','content':'Corrige ce code'},
            {'role':'assistant','content':'Réponse 1 non approuvée','learn':False},
            {'role':'user','content':'Réessaie'},
            {'role':'assistant','content':'Réponse corrigée','learn':True},
        ]}
        self.add('conversations/choix.jsonl', json.dumps(messages, ensure_ascii=False))
        mod.main([])
        conv = self.rows('conversations_train')[0]['messages']
        self.assertIs(conv[1]['learn'], False)
        self.assertIs(conv[3]['learn'], True)

    def test_secrets_and_binaries_excluded(self):
        self.add('danger/.env', 'DATABASE_PASSWORD=motdepasse')
        self.add('danger/cles.txt', 'API_KEY=sk-proj-example-token-is-sensitive')
        self.add('code/index.php', "<?php echo 'Salut';")
        self.add('code/tool.xyz', b'\x00\x01\x02')
        self.add('code/vendor/lib.php', "<?php echo 'DEPENDANCE';")
        mod.main([])
        report = json.loads((mod.OUTPUT / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(1, report['pretrain_documents'])
        self.assertGreaterEqual(report['skipped_count'], 2)
        self.assertNotIn('DEPENDANCE', self.pretrain()[0]['text'])

    def test_file_groups_do_not_leak_train_validation(self):
        for i in range(40):
            self.add(f'textes/phrase_{i}.txt', f'Phrase distincte numéro {i}.')
        mod.main([])
        train = self.rows('pretrain_train')
        val = self.rows('pretrain_validation')
        self.assertEqual(len(train) + len(val), 40)
        self.assertGreater(len(val), 0)
        self.assertLess(len(val), 40)
        self.assertFalse({x['text'] for x in train} & {x['text'] for x in val})

    def test_preserve_whitespace_python_when_split(self):
        code = ('def calcul(x):\n'
                '    valeur = x + 1\n'
                '    if valeur > 3:\n'
                '        return valeur\n'
                '    return 0\n') * 8
        path = self.add('code/indentation.py', code)
        mod.main(['--chunk-chars', '512'])
        docs = self.pretrain()
        pieces = []
        for row in docs:
            content = row['text'].split('Content:\n', 1)[1].rsplit('\n<|file_end|>', 1)[0]
            pieces.append(content)
        
        self.assertTrue(any('\n    valeur' in p for p in pieces))
        self.assertTrue(any(p.startswith(('    ', 'def ')) for p in pieces))
        self.assertEqual(path.read_text(encoding='utf-8').splitlines()[1], '    valeur = x + 1')

    def test_missing_legacy_monoliths(self):
        self.add('file.js', 'const nombre = 42;')
        mod.main([])
        self.assertFalse((mod.OUTPUT / 'pretrain_train.jsonl').exists())
        self.assertFalse((mod.OUTPUT / 'corpus_tokenizer.txt').exists())
        self.assertEqual(self.manifest()['version'], 2)


if __name__ == '__main__':
    unittest.main()
