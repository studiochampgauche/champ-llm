"""Tools for the champ-llm local training pipeline."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageDraw

from analyze_image import analyze
from train_vision_pytorch import parse_args, train
from vision_model import (VisionEncoder, image_to_tiles, assemble_example, generate_answer)
from vision_checkpoints import VisionShardStream, checkpoint_path, load_checkpoint, manifest_sha256
from vision_integrity import sha256_file


class MiniTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [4 + (ord(char) % 95) for char in text]

    def decode(self, seq, skip_special_tokens=True):
        return ''.join(chr(((int(i)-4)%95) + 32) for i in seq if int(i) != 1)


class MiniCausalLM(nn.Module):
    def __init__(self, vocab=110, hidden=16, max_context=1024):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden, max_position_embeddings=max_context, use_cache=False)
        self.embedding = nn.Embedding(vocab, hidden)
        self.rnn = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                labels=None, use_cache=False, past_key_values=None):
        tokens = self.embedding(input_ids) if inputs_embeds is None else inputs_embeds
        states, hidden = self.rnn(tokens, past_key_values)
        logits = self.head(states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        return SimpleNamespace(loss=loss, logits=logits,
                               past_key_values=hidden if use_cache else None)


class VisionCycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.images = self.root/'training_data'/'vision'
        self.images.mkdir(parents=True)
        im = Image.new('RGB', (400, 900), 'white')
        draw = ImageDraw.Draw(im)
        draw.rectangle((2, 2, 45, 40), fill='red')
        draw.rectangle((350, 840, 395, 890), fill='blue')
        self.image = self.images/'page.png'
        im.save(self.image)
        sidecar = self.image.with_suffix('.vision.json')
        sidecar.write_text(json.dumps({'image_sha256':sha256_file(self.image), 'critique':'Fond blanc'}),encoding='utf-8')
        self.model_dir = self.root/'model'
        self.model_dir.mkdir()
        (self.model_dir/'config.json').write_text('{"hidden_size":16,"max_position_embeddings":1024}')
        (self.model_dir/'tokenizer.json').write_text('{"vocab":"test-fixture"}')
        self.manifest_dir = self.root/'outputs'/'vision'
        data_folder = self.manifest_dir/'shards'/'vision_train'
        data_folder.mkdir(parents=True)
        for i in range(2):
            rows = [json.dumps({'image':'training_data/vision/page.png',
                                'image_sha256':sha256_file(self.image),
                                'annotation':'training_data/vision/page.vision.json',
                                'annotation_sha256':sha256_file(sidecar),
                                'question': f'Analyse cet écran {i}',
                                'answer': f'Le fond est blanc. Le bouton doit être visible {i}.'},
                               ensure_ascii=False)]
            (data_folder/f'shard-{i:05d}.jsonl').write_text('\n'.join(rows)+'\n',encoding='utf-8')
        self.manifest = self.manifest_dir/'vision_manifest.json'
        self.manifest.write_text(json.dumps({'version':1,'splits':{
            'vision_train':{'files':[{'path':f'shards/vision_train/shard-{i:05d}.jsonl'} for i in range(2)]},
            'vision_validation':{'files':[]}}}),encoding='utf-8')

    def tearDown(self):
        self.tmp.cleanup()

    def test_tiles_preserve_detail_beyond_small_overview(self):
        tiles = image_to_tiles(self.image, 32, max_tiles=5)
        self.assertEqual(5, tiles.shape[0])
        self.assertEqual((3,32,32), tuple(tiles.shape[1:]))
        
        self.assertGreater(tiles[-1, 2].mean().item(), .08)
        self.assertFalse(torch.equal(tiles[0],tiles[-1]))

    def test_large_answer_fails_loudly_instead_of_silent_cut(self):
        tok = MiniTokenizer()
        long_answer = 'C' * 1000
        with self.assertRaisesRegex(ValueError, 'truncated'):
            assemble_example(tok, 'Question', long_answer, max_answer=128)
        ids, labels = assemble_example(tok, 'Question', long_answer, max_answer=128, allow_truncation=True)
        self.assertEqual(int((labels != -100).sum()), 128)

    def test_shard_cursor_exact_without_loading_corpus_in_memory(self):
        s = VisionShardStream(self.manifest, self.root, seed=42)
        first = next(s)[1]['question']
        cursor = dict(s.cursor)
        second = next(s)[1]['question']
        s2 = VisionShardStream(self.manifest, self.root, seed=42, cursor=cursor)
        self.assertEqual(second, next(s2)[1]['question'])
        self.assertNotEqual(first, second)

    def _opts(self, output, steps, *additional):
        return parse_args(['--model',str(self.model_dir),'--manifest',str(self.manifest),
                           '--output',str(output),'--steps',str(steps),
                           '--image-size','32','--max-tiles','2','--pool-grid','1',
                           '--vision-dim','16','--vision-layers','1',
                           '--max-prompt-tokens','128','--max-answer-tokens','150',
                           '--save-every','1',*additional])

    @staticmethod
    def mini_loader(source, device, dtype):
        return MiniCausalLM().to(device), MiniTokenizer()

    def test_full_cycle_resume_optimizer_and_analyze_image(self):
        
        from importlib import import_module
        trainer = import_module('train_vision_pytorch')
        with patch.object(trainer, 'ROOT', self.root):
            train(self._opts(self.root/'continuous',3),load_language=self.mini_loader)
            
            partial_out = self.root/'repris'
            train(self._opts(partial_out,2),load_language=self.mini_loader)
            train(self._opts(partial_out,1,'--resume'),load_language=self.mini_loader)
        ckpt_full = checkpoint_path(self.root/'continuous',resume=True)
        ckpt_resume = checkpoint_path(partial_out,resume=True)
        a = torch.load(ckpt_full/'state.pt', map_location='cpu', weights_only=True)
        b = torch.load(ckpt_resume/'state.pt', map_location='cpu', weights_only=True)
        self.assertEqual(json.loads((ckpt_full/'metadata.json').read_text())['step'],3)
        self.assertEqual(json.loads((ckpt_resume/'metadata.json').read_text())['step'],3)
        
        from safetensors.torch import load_file
        weights_a,weights_b=load_file(str(ckpt_full/'vision.safetensors')),load_file(str(ckpt_resume/'vision.safetensors'))
        for k in weights_a:
            self.assertTrue(torch.equal(weights_a[k], weights_b[k]),k)
        
        self.assertGreater(len(b['optimizer']['state']), 0)
        meta = json.loads((ckpt_resume/'metadata.json').read_text(encoding='utf-8'))
        self.assertEqual(meta['cursor']['count'],3)
        self.assertEqual(meta['manifest_sha256'],manifest_sha256(self.manifest))
        
        torch.manual_seed(3407)
        mini = MiniCausalLM()
        response = analyze(ckpt_resume,self.image,'Décris cette capture',mini,MiniTokenizer(),max_new_tokens=5)
        self.assertIsInstance(response,str)
        self.assertTrue(len(response) >= 1)

    def test_resume_rejects_modified_manifest(self):
        from importlib import import_module
        trainer=import_module('train_vision_pytorch')
        out=self.root/'experiences'
        with patch.object(trainer,'ROOT',self.root):
            train(self._opts(out,1),load_language=self.mini_loader)
            self.manifest.write_text(self.manifest.read_text()+'\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'manifest'):
                train(self._opts(out,1,'--resume'),load_language=self.mini_loader)


if __name__ == '__main__':
    unittest.main()
