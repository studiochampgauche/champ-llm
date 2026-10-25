"""Tools for the champ-llm local training pipeline."""
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from local_backend import ChampBackend, verify_local_model, resolve_vision_checkpoint
import coding_agent


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [ord(c) + 10 for c in text]

    def get_vocab(self):
        return {'<|end|>': 3}

    def decode(self, values, *, skip_special_tokens):
        return '{"edits":[],"comment":"rien à modifier"}'


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x = torch.nn.Parameter(torch.ones((1,)))
        self.config = types.SimpleNamespace(max_position_embeddings=1000)

    def generate(self, *, input_ids, **kwargs):
        self.last_input_ids = input_ids.detach().clone()
        self.last_options = kwargs
        return torch.cat([input_ids, torch.tensor([[4, 5]], device=input_ids.device)], dim=1)


class FakeBackend:
    has_vision = True
    def generate_code(self, prompt):
        assert 'GOAL:' in prompt
        return json.dumps({'edits': [], 'comment': 'Pas de correction'})

    def critique_images(self, paths):
        return 'Contraste insuffisant sur mobile'


class BackendLocalTests(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name)

    def test_verify_local_model(self):
        model = self.root / 'weights'
        model.mkdir()
        with self.assertRaises(FileNotFoundError):
            verify_local_model(model)
        (model/'config.json').write_text('{}')
        (model/'tokenizer.json').write_text('{}')
        (model/'model-00001.safetensors').write_bytes(b'fake')
        self.assertEqual(verify_local_model(model), model.resolve())

    def test_resolve_checkpoint(self):
        ckpt = self.root / 'run' / 'checkpoints' / 'step-00000002'
        ckpt.mkdir(parents=True)
        (ckpt/'vision.safetensors').write_bytes(b'dummy')
        (ckpt/'vision_config.json').write_text('{}')
        (ckpt/'state.pt').write_bytes(b'dummy')
        (self.root/'run'/'latest.json').write_text(json.dumps({'checkpoint': ckpt.name, 'step': 2}))
        self.assertEqual(resolve_vision_checkpoint(self.root/'run'), ckpt.resolve())

    def test_direct_generation_without_ollama(self):
        backend = ChampBackend.__new__(ChampBackend)
        backend.device = torch.device('cpu')
        backend.model = FakeModel().eval()
        backend.tokenizer = FakeTokenizer()
        backend.max_generation = 50
        self.assertIn('comment', backend.generate_code('un exemple'))
        from text_training_common import conversation_tokens, assistant_prompt_tokens
        prefix = assistant_prompt_tokens(backend.tokenizer, 'un exemple')
        self.assertEqual(backend.model.last_input_ids[0].tolist(), prefix)
        full, _ = conversation_tokens(backend.tokenizer, [
            {'role':'user', 'content':'un exemple'},
            {'role':'assistant', 'content':'voici une réponse'}])
        self.assertEqual(full[:len(prefix)], prefix)  
        self.assertEqual(backend.model.last_options['eos_token_id'], [2, 3])
        self.assertNotIn('Utilisateur:', 'un exemple')
        backend.model.config.max_position_embeddings = 52
        with self.assertRaisesRegex(ValueError, 'Prompt too long'):
            backend.generate_code('un exemple')
        with self.assertRaisesRegex(ValueError, 'nonempty text'):
            backend.generate_code('  ')

    def test_direct_agent_does_not_call_ollama(self):
        project = self.root / 'site'
        project.mkdir()
        (project/'index.html').write_text('<h1>Bonjour</h1>')
        with patch.object(coding_agent, 'ROOT', self.root), \
             patch.object(coding_agent, 'inspect_site', return_value={'results': []}), \
             patch.object(coding_agent, 'ollama_chat', side_effect=AssertionError('Ollama interdit')):
            report = coding_agent.run_agent(project, 'http://localhost:3000', 'Un beau site',
                                            code_model='champ-llm', visual_model=None,
                                            test_command=None, interactions=None, apply=False,
                                            iterations=1, local_backend=FakeBackend())
        self.assertEqual(report['backend'], 'champ-llm')
        self.assertEqual(report['cycles'][0]['model_comment'], 'Pas de correction')

    def test_local_visual_feedback_uses_adapter(self):
        project = self.root / 'capture'
        project.mkdir()
        image_file = project / 'image.png'
        image_file.write_bytes(b'not a real image, but path metadata is sufficient for this test')
        report = {'results': [{'capture': 'image.png'}]}
        with patch.object(coding_agent, 'ollama_chat', side_effect=AssertionError('Ollama interdit')):
            critique = coding_agent.visual_feedback(project, report, None, local_backend=FakeBackend())
        self.assertIn('Contraste', critique)


if __name__ == '__main__':
    unittest.main()
