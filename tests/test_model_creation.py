"""Exercise size planning, real model creation and training across architectures."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import create_model
from model_architecture import Architecture, PRESETS, llama_parameter_count


class ModelPlanningTests(unittest.TestCase):
    def test_plan_needs_no_corpus_tokenizer_ml_packages_or_file_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('create_model.py', 'model_architecture.py'):
                (root / name).write_bytes((create_model.ROOT / name).read_bytes())
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            result = subprocess.run([sys.executable, '-B', '-S', str(root / 'create_model.py'),
                                     '--size', '7b', '--plan'], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('7,002,705,920', result.stdout)
            self.assertIn('provisional', result.stdout)
            self.assertIn('no files or weights created', result.stdout)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_explicit_size_and_consistent_options_are_required(self):
        for argv in (['--plan'], ['--create'], ['--plan', '--size', 'unknown'],
                     ['--plan', '--size', '7b', '--config', 'custom.json'],
                     ['--create', '--size', '7b', '--vocab-size', '32768'],
                     ['--plan', '--size', '7b', '--vocab-size', '0'],
                     ['--plan', '--size', '7b', '--vocab-size', '32768', '--tokenizer', 'model']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                create_model.parse_args(argv)
            self.assertEqual(error.exception.code, 2)

    def test_invalid_custom_dimensions_are_rejected(self):
        valid = Architecture(64, 128, 2, 4, 2).to_dict()
        for change in ({'hidden_size': 63}, {'hidden_size': 60}, {'num_key_value_heads': 3},
                       {'num_hidden_layers': 0}, {'intermediate_size': 1.5},
                       {'num_attention_heads': True}, {'max_position_embeddings': -1},
                       {'typo': 123}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                Architecture.from_dict({**valid, **change})
        for data in ([], {}, {'hidden_size': 64}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                Architecture.from_dict(data)
        with self.assertRaises(ValueError):
            PRESETS['7b'].parameter_count(-1)

    def test_bad_json_and_missing_explicit_tokenizer_fail_without_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / 'bad.json'
            config.write_text('{broken', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                create_model.parse_args(['--plan', '--config', str(config)])
            with self.assertRaisesRegex(SystemExit, 'Tokenizer missing'):
                create_model.main(['--plan', '--size', '1b', '--tokenizer', str(root / 'missing'),
                                   '--directory', str(root / 'out')])
            self.assertFalse((root / 'out').exists())

    def test_preset_counts_match_real_llama_shapes_without_allocating_weights(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        for name, architecture in PRESETS.items():
            for vocab in (257, 32768):
                with self.subTest(size=name, vocab=vocab), torch.device('meta'):
                    model = LlamaForCausalLM(LlamaConfig(**architecture.to_dict(), vocab_size=vocab,
                                                        tie_word_embeddings=False,
                                                        attention_bias=False, mlp_bias=False))
                    self.assertEqual(architecture.parameter_count(vocab), sum(p.numel() for p in model.parameters()))
                    self.assertTrue(all(p.is_meta for p in model.parameters()))
            target = float(name[:-1]) * (1e9 if name.endswith('b') else 1e6)
            self.assertLess(abs(architecture.parameter_count(32768) - target) / target, 0.03)


    def test_training_count_handles_tied_embeddings_biases_and_head_dimensions(self):
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        for tied in (True, False):
            with self.subTest(tied=tied), torch.device('meta'):
                config = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=300,
                    tie_word_embeddings=tied, attention_bias=True, mlp_bias=True)
                model = LlamaForCausalLM(config)
                self.assertEqual(llama_parameter_count(config.to_dict()), sum(p.numel() for p in model.parameters()))


class ModelCreationIntegrationTests(unittest.TestCase):
    def test_custom_models_create_train_resume_and_preserve_source_tokenizer(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
        from train_text import parse_args, run_training
        from local_backend import ChampBackend
        from safetensors.torch import load_file
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'tokenizer'
            raw = Tokenizer(models.BPE(unk_token='<unk>'))
            raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            raw.decoder = decoders.ByteLevel()
            specials = ['<pad>', '<s>', '</s>', '<unk>', '<|system|>', '<|user|>', '<|assistant|>', '<|tool|>', '<|end|>']
            raw.train_from_iterator(['Bonjour! Hello! function test() { return 1; }'] * 3,
                                    trainers.BpeTrainer(vocab_size=280, special_tokens=specials,
                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>', bos_token='<s>',
                eos_token='</s>', unk_token='<unk>', additional_special_tokens=specials[4:])
            tokenizer.save_pretrained(source)
            original_tokenizer = {p.name: p.read_bytes() for p in source.iterdir()}
            occupied = root / 'occupied'
            occupied.mkdir()
            (occupied / 'notes.txt').write_text('Keep my files', encoding='utf-8')
            with self.assertRaisesRegex(SystemExit, 'must be empty'):
                create_model.main(['--size', '7b', '--create', '--tokenizer', str(source), '--directory', str(occupied)])
            self.assertEqual([p.name for p in occupied.iterdir()], ['notes.txt'])
            self.assertEqual((occupied / 'notes.txt').read_text(), 'Keep my files')
            shard = root / 'data.jsonl'
            shard.write_text(json.dumps({'text': 'Bonjour! Hello! Code and design. ' * 30}) + '\n', encoding='utf-8')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'version': 2, 'splits': {'pretrain_train': {'files': [{'path': shard.name}]}}}), encoding='utf-8')
            for hidden, layers in ((32, 1), (64, 2)):
                with self.subTest(hidden=hidden, layers=layers):
                    architecture = Architecture(hidden, hidden * 3, layers, 4, 2, 128)
                    config_path = root / 'architecture.json'
                    config_path.write_text(json.dumps(architecture.to_dict()), encoding='utf-8')
                    output = root / f'model-{hidden}'
                    options = ['--config', str(config_path), '--tokenizer', str(source), '--directory', str(output)]
                    with contextlib.redirect_stdout(io.StringIO()) as planned:
                        create_model.main(options + ['--plan'])
                    self.assertFalse(output.exists())
                    self.assertIn(f'{architecture.parameter_count(len(tokenizer)):,}', planned.getvalue())
                    create_model.main(options + ['--create'])
                    identity = json.loads((output / 'model_identity.json').read_text())
                    self.assertEqual(identity['name'], 'champ-llm')
                    self.assertEqual(identity['profile'], 'custom')
                    self.assertEqual(identity['params'], architecture.parameter_count(len(tokenizer)))
                    initial_weights = load_file(str(output / 'model.safetensors'))
                    self.assertEqual(identity['params'], sum(p.numel() for p in initial_weights.values()))
                    self.assertEqual(AutoTokenizer.from_pretrained(output).model_max_length, 128)
                    snapshot = {p.name: p.read_bytes() for p in output.iterdir()}
                    with self.assertRaisesRegex(SystemExit, 'already exists'):
                        create_model.main(options + ['--create'])
                    self.assertEqual(snapshot, {p.name: p.read_bytes() for p in output.iterdir()})
                    run = root / f'run-{hidden}'
                    common = ['--model', str(output), '--data', str(manifest), '--output', str(run),
                              '--context', '16', '--allow-cpu']
                    with patch.object(torch.cuda, 'is_available', return_value=False):
                        run_training(parse_args(common + ['--steps', '1']))
                        run_training(parse_args(common + ['--steps', '2', '--resume']))
                    latest = run / 'checkpoints/step-00000002/model'
                    learned = load_file(str(latest / 'model.safetensors'))
                    self.assertTrue(any(not torch.equal(initial_weights[k], learned[k]) for k in initial_weights))
                    self.assertTrue(all(torch.isfinite(w).all().item() for w in learned.values()))
                    model = AutoModelForCausalLM.from_pretrained(latest)
                    self.assertEqual(model.config.hidden_size, hidden)
                    self.assertEqual(model.config.num_hidden_layers, layers)
                    backend = ChampBackend(latest, device='cpu', max_generation=2)
                    self.assertIsInstance(backend.generate_code('Hello'), str)
            self.assertEqual(original_tokenizer, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_existing_model_rejected_before_loading_tokenizer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'existing'
            output.mkdir()
            (output / 'config.json').write_text('{"owned": true}', encoding='utf-8')
            with self.assertRaisesRegex(SystemExit, 'already exists'):
                create_model.main(['--create', '--size', '7b', '--directory', str(output)])
            self.assertEqual((output / 'config.json').read_text(), '{"owned": true}')


if __name__ == '__main__':
    unittest.main()
