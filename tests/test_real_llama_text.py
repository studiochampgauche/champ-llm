"""Train and resume a real tiny Llama without pretrained weights."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HAVE_HF = all(importlib.util.find_spec(name) is not None for name in ('transformers','tokenizers','safetensors'))


@unittest.skipUnless(HAVE_HF, 'Install transformers, tokenizers and safetensors to test a real Llama')
class RealLlamaTextTests(unittest.TestCase):
    def test_pretrain_sft_resume(self):
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from train_text import parse_args, run_training
        from safetensors.torch import load_file
        import torch

        def train(options):
            # Exact resumption is tested on CPU, even when the suite runs on a GPU host.
            with patch.object(torch.cuda, 'is_available', return_value=False):
                run_training(parse_args(options))

        def assert_state_equal(left, right, name='state'):
            if isinstance(left, torch.Tensor):
                self.assertTrue(torch.equal(left, right), name)
            elif isinstance(left, dict):
                self.assertEqual(left.keys(), right.keys(), name)
                for key in left:
                    assert_state_equal(left[key], right[key], f'{name}.{key}')
            elif isinstance(left, (list, tuple)):
                self.assertEqual(type(left), type(right), name)
                self.assertEqual(len(left), len(right), name)
                for index, (a, b) in enumerate(zip(left, right)):
                    assert_state_equal(a, b, f'{name}[{index}]')
            else:
                self.assertEqual(left, right, name)

        with tempfile.TemporaryDirectory(prefix='champ-text-llama-') as temporary:
            root = Path(temporary)
            model_dir = root/'model'
            model_dir.mkdir()
            tokenizer_impl = Tokenizer(models.BPE(unk_token='<unk>'))
            tokenizer_impl.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer_impl.decoder = decoders.ByteLevel()
            special = ['<pad>','<s>','</s>','<unk>','<|system|>','<|user|>','<|assistant|>','<|tool|>','<|end|>']
            corpus = ['Bonjour, aide-moi à programmer en HTML, CSS et React.',
                      'function Composant(){return <h1>Salut</h1>}',
                      'Salut, comment vas-tu? Je peux te répondre en français.']
            tokenizer_impl.train_from_iterator(corpus * 12, trainers.BpeTrainer(
                vocab_size=350, min_frequency=1, special_tokens=special,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_impl,
                        pad_token='<pad>', bos_token='<s>', eos_token='</s>', unk_token='<unk>',
                        additional_special_tokens=special[4:])
            tokenizer.save_pretrained(model_dir)
            torch.manual_seed(123)
            model = LlamaForCausalLM(LlamaConfig(
                vocab_size=len(tokenizer), hidden_size=48, intermediate_size=96,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=256, attention_dropout=0.1,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id))
            model.save_pretrained(model_dir, safe_serialization=True)
            del model
            output = root/'outputs'
            splits = {}
            for name, rows in {
                'pretrain_train':[{'text':'Bonjour et React, PHP, HTML et JavaScript. ' * 15}],
                'conversations_train':[{'messages':[
                    {'role':'user','content':'Écris du HTML.'},
                    {'role':'assistant','content':'<h1>Bonjour</h1>'}]}],
            }.items():
                shard = output/'shards'/name/'shard-00000.jsonl'
                shard.parent.mkdir(parents=True, exist_ok=True)
                shard.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
                splits[name]={'files':[{'path':f'shards/{name}/{shard.name}'}]}
            manifest=output/'shards_manifest.json'
            manifest.write_text(json.dumps({'version':2,'splits':splits}), encoding='utf-8')
            source = model_dir
            for mode in ('pretrain', 'conversations'):
                dest = root/'runs'/mode
                reference = root/'runs'/(mode + '_continuous')
                initial_weights = load_file(str(source/'model.safetensors'))
                common = ['--model', str(source), '--data', str(manifest),
                          '--context', '32', '--mode', mode, '--allow-cpu',
                          '--gradient-accumulation', '2', '--save-every', '1']
                train(common + ['--output', str(dest), '--steps', '1'])
                first = dest/'checkpoints'/'step-00000001'
                self.assertTrue((first/'model'/'config.json').exists())
                learned = load_file(str(first/'model'/'model.safetensors'))
                self.assertTrue(all(torch.isfinite(w).all().item() for w in learned.values()))
                self.assertTrue(any(not torch.equal(initial_weights[k], learned[k])
                                    for k in initial_weights), 'Training must change the weights')
                train(common + ['--output', str(dest), '--steps', '2', '--resume'])
                latest = json.loads((dest/'latest.json').read_text(encoding='utf-8'))
                self.assertEqual(latest['step'], 2)
                train(common + ['--output', str(reference), '--steps', '2'])
                resumed = dest/'checkpoints'/'step-00000002'
                continuous = reference/'checkpoints'/'step-00000002'
                resumed_weights = load_file(str(resumed/'model'/'model.safetensors'))
                continuous_weights = load_file(str(continuous/'model'/'model.safetensors'))
                assert_state_equal(continuous_weights, resumed_weights, mode + '.weights')
                resumed_state = torch.load(resumed/'state.pt', map_location='cpu', weights_only=True)
                continuous_state = torch.load(continuous/'state.pt', map_location='cpu', weights_only=True)
                self.assertTrue(resumed_state['optimizer']['state'])
                assert_state_equal(continuous_state, resumed_state, mode + '.resume')
                # Start conversation SFT from the actual pretrained checkpoint.
                source = resumed/'model'
            print('TEXT VALIDATED: weights changed, pretrain -> SFT, exact resumption '
                  '(weights, optimizer, RNG and cursor), with dropout and accumulation.')



if __name__ == '__main__':
    unittest.main()
