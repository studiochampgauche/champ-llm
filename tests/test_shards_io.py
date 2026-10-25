"""Tools for the champ-llm local training pipeline."""
from __future__ import annotations
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shards_io import resolve_shards


class ShardPathTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'shards' / 'pretrain_train').mkdir(parents=True)
        for i in range(3):
            (self.root / 'shards' / 'pretrain_train' / f'shard-{i:05d}.jsonl').write_text(
                json.dumps({'text': f'Texte {i}'}) + '\n', encoding='utf-8'
            )
        self.data = {'version': 2, 'splits': {
            'pretrain_train': {'files': [{'path': f'shards/pretrain_train/shard-{i:05d}.jsonl'}
                                           for i in range(3)]},
            'tokenizer': {'files': []},
        }}
        self.manifest = self.root / 'shards_manifest.json'
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_manifest_all_shards_in_order(self):
        paths = resolve_shards(self.manifest)
        self.assertEqual(len(paths), 3)
        self.assertTrue(paths[2].name.endswith('00002.jsonl'))

    def test_directory_mode(self):
        paths = resolve_shards(self.root / 'shards')
        self.assertEqual(len(paths), 3)

    def test_traversal_rejected(self):
        self.data['splits']['pretrain_train']['files'][0]['path'] = '../../outside.jsonl'
        self.manifest.write_text(json.dumps(self.data), encoding='utf-8')
        with self.assertRaises(ValueError):
            resolve_shards(self.manifest)

    def test_legacy_single_jsonl(self):
        path = self.root / 'single.jsonl'
        path.write_text('{"text":"bonjour"}\n', encoding='utf-8')
        self.assertEqual(resolve_shards(path), [path.resolve()])

    def test_pytorch_generator_iterates_shards(self):
        
        import train_pytorch
        class DummyTokenizer:
            eos_token_id = 0
            def encode(self, text, add_special_tokens=False):
                return [ord(x) for x in text]
        chunks = list(train_pytorch.chunks_from_jsonl(resolve_shards(self.manifest), DummyTokenizer(), 3))
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(chunk) == 4 for chunk in chunks))


if __name__ == '__main__':
    unittest.main()
