"""Tools for the champ-llm local training pipeline."""
from __future__ import annotations
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prepare_data_large', HERE / 'prepare_data.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class LargeDataTests(unittest.TestCase):
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

    def paths(self, split):
        manifest = json.loads((mod.OUTPUT / 'shards_manifest.json').read_text(encoding='utf-8'))
        return [mod.OUTPUT / item['path'] for item in manifest['splits'][split]['files']]

    def rows(self, split):
        return [json.loads(line) for path in self.paths(split)
                for line in path.read_text(encoding='utf-8').splitlines() if line]

    def test_large_php_with_split_on_one_side(self):
        path = mod.SOURCES / 'php' / 'grand_script.php'
        path.parent.mkdir()
        with path.open('w', encoding='utf-8') as f:
            f.write('<?php\n')
            for i in range(90_000):
                f.write(f'    $ligne_{i} = "Bonjour num {i}";\n')
        self.assertGreater(path.stat().st_size, 2 * 1024**2)
        mod.main(['--max-mb', '6', '--chunk-chars', '4096', '--shard-mb', '.2'])
        train, val = self.rows('pretrain_train'), self.rows('pretrain_validation')
        all_rows = train + val
        self.assertGreater(len(all_rows), 50)
        self.assertFalse(train and val, 'One source file must not occur in both splits')
        self.assertTrue(all('Language: PHP' in item['text'] for item in all_rows))
        self.assertGreater(len(self.paths('pretrain_train')) + len(self.paths('pretrain_validation')), 2)
        self.assertFalse(list(mod.OUTPUT.glob('.preparation_*')))

    def test_jsonl_text_record_segments_stay_in_one_partition(self):
        source = mod.SOURCES / 'corpus.jsonl'
        with source.open('w', encoding='utf-8') as f:
            for i in range(100):
                text = (f'DOCUMENT_{i:03d} :: ' + ('A' if i % 2 else 'B') +
                        f' "lorem ipsum_{i:03d}"\n') * 80
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
        mod.main(['--chunk-chars', '512', '--shard-mb', '.02'])
        train, val = self.rows('pretrain_train'), self.rows('pretrain_validation')
        self.assertTrue(train and val)
        def doc_ids(rows):
            ids = set()
            for row in rows:
                import re
                for marker in re.findall(r'DOCUMENT_\d{3}', row['text']):
                    ids.add(marker)
            return ids
        self.assertFalse(doc_ids(train) & doc_ids(val), 'JSONL segments from one document leaked across splits')
        self.assertGreater(len(self.paths('pretrain_train')), 1)

    def test_sft_pretrain_alignment(self):
        path = mod.SOURCES / 'questions.jsonl'
        with path.open('w', encoding='utf-8') as f:
            for i in range(120):
                msg = {'messages': [
                    {'role': 'user', 'content': f'Question #{i}'},
                    {'role': 'assistant', 'content': f'Réponse #{i}'}]}
                f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        mod.main(['--shard-mb', '.01'])
        sft_valid = self.rows('conversations_validation')
        pretrain_train = {x['text'] for x in self.rows('pretrain_train')}
        pretrain_valid = {x['text'] for x in self.rows('pretrain_validation')}
        self.assertTrue(sft_valid)
        for row in sft_valid:
            rendered = mod.conversations_to_text(row['messages'])
            self.assertNotIn(rendered, pretrain_train)
            self.assertIn(rendered, pretrain_valid)

    def test_size_limit_is_configurable(self):
        (mod.SOURCES / 'x.txt').write_text('Bonjour\n', encoding='utf-8')
        big = mod.SOURCES / 'long.css'
        with big.open('w', encoding='utf-8') as f:
            for i in range(130_000):
                f.write(f'.classe-{i} {{ color: red; }}\n')
        self.assertGreater(big.stat().st_size, 2 * 1024**2)
        mod.main(['--max-mb', '2'])
        report = json.loads((mod.OUTPUT / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['source_files_read'], 1)
        self.assertTrue(any('file too large' in x['reason'] for x in report['skipped_files']))
        mod.main(['--max-mb', '10'])
        report = json.loads((mod.OUTPUT / 'report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['source_files_read'], 2)

    def test_indent_exact_across_chunks(self):
        code = ''.join(f'def foo_{i}():\n    return {i}\n' for i in range(150))
        p = mod.SOURCES / 'code.py'
        for newline in ('\n', '\r\n'):
            with self.subTest(newline=repr(newline)):
                expected = code.replace('\n', newline)
                # Write exact bytes: write_text would translate LF to CRLF on Windows.
                p.write_bytes(expected.encode('utf-8'))
                mod.main(['--chunk-chars', '512', '--shard-mb', '.002'])
                text = ''.join(x['text'].split('Content:\n', 1)[1].rsplit('\n<|file_end|>', 1)[0]
                               for x in self.rows('pretrain_train') + self.rows('pretrain_validation'))
                self.assertEqual(text, expected)

    def test_single_very_large_line_is_streamed(self):
        import tracemalloc
        p = mod.SOURCES / 'huge_custom.xyz'
        
        with p.open('w', encoding='utf-8') as f:
            for i in range(1200):
                f.write(f'{i:08d} ' + 'abcde' * 1995)
        self.assertGreater(p.stat().st_size, 10 * 1024**2)
        tracemalloc.start()
        try:
            mod.main(['--max-mb', '20', '--chunk-chars', '4096', '--shard-mb', '1'])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        
        self.assertLess(peak, 8 * 1024**2, f'Peak Python memory: {peak / 1024**2:.1f} MiB')
        self.assertGreater(len(self.rows('pretrain_train') + self.rows('pretrain_validation')), 1)

    def test_jsonl_oversized_record_does_not_break_other_records(self):
        path = mod.SOURCES / 'data.jsonl'
        with path.open('w', encoding='utf-8') as f:
            f.write(json.dumps({'text': 'A' * 160000}) + '\n')
            f.write(json.dumps({'text': 'Conservé après une line énorme'}) + '\n')
        mod.main(['--max-record-mb', '.05'])
        full = self.rows('pretrain_train') + self.rows('pretrain_validation')
        self.assertEqual(len(full), 1)
        self.assertIn('Conservé après', full[0]['text'])

if __name__ == '__main__':
    unittest.main()
