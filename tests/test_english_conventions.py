"""Regression tests for English developer interfaces and multilingual content."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_data
import train_text
from capture_website import create_annotation


class EnglishConventionTests(unittest.TestCase):
    def test_cli_flags_are_english(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'new_training_run'
            args = train_text.parse_args([
                '--engine', 'pytorch', '--model', str(Path(temp) / 'model'),
                '--output', str(output), '--steps', '2',
                '--context', '128', '--gradient-accumulation', '2',
            ])
            self.assertEqual(args.steps, 2)
            self.assertEqual(args.context, 128)
            self.assertEqual(args.gradient_accumulation, 2)

    def test_french_text_is_preserved_but_metadata_is_english(self):
        with tempfile.TemporaryDirectory() as temp:
            home = Path(temp)
            dataset = home / 'training_data'
            source = dataset / 'programming' / 'php' / 'greet.php'
            source.parent.mkdir(parents=True)
            source.write_text('<?php echo "Bonjour, bienvenue !";', encoding='utf-8')
            conversations = dataset / 'french' / 'conversation.jsonl'
            conversations.parent.mkdir(parents=True)
            conversations.write_text(json.dumps({
                'messages': [
                    {'role': 'user', 'content': 'Bonjour, comment vas-tu ?'},
                    {'role': 'assistant', 'content': 'Je vais bien, merci.', 'learn': True},
                ]
            }, ensure_ascii=False) + '\n', encoding='utf-8')
            with patch.object(prepare_data, 'ROOT', home), \
                 patch.object(prepare_data, 'SOURCES', dataset), \
                 patch.object(prepare_data, 'SINGLE', home / 'training_data.txt'), \
                 patch.object(prepare_data, 'OUTPUT', home / 'outputs'):
                prepare_data.main([])
            manifest = json.loads((home / 'outputs' / 'shards_manifest.json').read_text())
            paths = [home / 'outputs' / row['path'] for row in manifest['splits']['pretrain_train']['files']]
            text = ''.join(p.read_text(encoding='utf-8') for p in paths)
            self.assertIn('Language: PHP', text)
            self.assertIn('Path: training_data/programming/php/greet.php', text)
            self.assertIn('Bonjour', text)
            self.assertTrue((home / 'outputs' / 'shards_manifest.json').exists())

    def test_english_annotation_keys(self):
        with tempfile.TemporaryDirectory() as temp:
            image = Path(temp) / 'sample.png'
            image.write_bytes(b'sample bytes for hash check')
            annotation = create_annotation(image, project='fixture-site', width=1440)
            record = json.loads(annotation.read_text(encoding='utf-8'))
            self.assertEqual(record['project'], 'fixture-site')
            self.assertEqual(record['viewport_width'], 1440)
            self.assertEqual(record['code_files'], [])
            self.assertIn('image_sha256', record)
            self.assertNotIn('projet', record)


if __name__ == '__main__':
    unittest.main()
