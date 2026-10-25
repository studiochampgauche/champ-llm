"""Tools for the champ-llm local training pipeline."""
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from text_training_common import (
    ShardTextStream, conversation_tokens, assistant_prompt_tokens, save_full_checkpoint,
    resolve_checkpoint, load_full_checkpoint,
)


class CharTokenizer:
    bos_token_id = 2
    eos_token_id = 1
    def encode(self, text, add_special_tokens=False):
        return [ord(c) + 4 for c in text]
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path)/'tokenizer.json').write_text('t')


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(300, 12)
        self.linear = nn.Linear(12, 300)
    def forward(self, ids):
        return self.linear(self.embedding(ids))
    def save_pretrained(self, path, **kwargs):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path)/'tiny_state.pt')


def fixture(tmp: Path, mode: str, lines: list[dict]):
    split = 'pretrain_train' if mode == 'pretrain' else 'conversations_train'
    output = tmp/'outputs'
    shard = output/'shards'/split/'shard-00000.jsonl'
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in lines), encoding='utf-8')
    manifest = output/'shards_manifest.json'
    manifest.write_text(json.dumps({'version':2,'splits':{
        split:{'files':[{'path':f'shards/{split}/{shard.name}'}]}}}), encoding='utf-8')
    return manifest, shard


class TextTrainingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.tokenizer = CharTokenizer()

    def test_role_mask_assistant_only(self):
        ids, labels = conversation_tokens(self.tokenizer, [
            {'role':'system','content':'Sois utile'},
            {'role':'user','content':'Dis-moi bonjour'},
            {'role':'assistant','content':'Bonjour'}])
        target = ''.join(chr(n-4) for n in labels if n >= 4)
        self.assertIn('Bonjour', target)
        self.assertNotIn('Dis-moi bonjour', target)
        self.assertEqual(len(ids), len(labels))
        self.assertTrue(any(x == -100 for x in labels))

    def test_inference_prefix_matches_sft(self):
        messages = [{'role': 'user', 'content': 'Explique PHP & HTML.'},
                    {'role': 'assistant', 'content': 'PHP s’exécute côté serveur.'}]
        ids_train, _ = conversation_tokens(self.tokenizer, messages)
        prefix = assistant_prompt_tokens(self.tokenizer, messages[0]['content'])
        self.assertEqual(prefix, ids_train[:len(prefix)])
        self.assertNotIn(self.tokenizer.eos_token_id, prefix)
        self.assertEqual(prefix[:1], [self.tokenizer.bos_token_id])
        self.assertEqual(prefix[-len(self.tokenizer.encode('<|assistant|>\n')):],
                         self.tokenizer.encode('<|assistant|>\n'))

    def test_multiple_answers_opt_out(self):
        ids, labels = conversation_tokens(self.tokenizer, [
            {'role':'user','content':'Salut'},
            {'role':'assistant','content':'Mauvais exemple','learn':False},
            {'role':'user','content':'Encore'},
            {'role':'assistant','content':'Bon exemple','learn':True}])
        target = ''.join(chr(n-4) for n in labels if n >= 4)
        self.assertNotIn('Mauvais', target)
        self.assertIn('Bon exemple', target)
        with self.assertRaises(ValueError):
            conversation_tokens(self.tokenizer,[{'role':'user','content':'hi'},
                                                     {'role':'assistant','content':'non','learn':False}])

    def test_stream_cursor_resumes_exactly(self):
        manifest, _ = fixture(self.base, 'pretrain', [{'text':'abcde ' * 300}, {'text':'zyx ' * 200}])
        s = ShardTextStream(manifest, self.tokenizer, mode='pretrain', context=32)
        _ = next(s)
        state = s.state_dict()
        same = ShardTextStream(manifest, self.tokenizer, mode='pretrain', context=32, cursor=state)
        self.assertEqual(next(s), next(same))
        self.assertEqual(next(s), next(same))

    def test_conversation_shards_mask(self):
        manifest, _ = fixture(self.base, 'conversations', [
            {'messages':[{'role':'user','content':'J'}, {'role':'assistant','content':'K'}]}])
        s = ShardTextStream(manifest,self.tokenizer,mode='conversations',context=64)
        ids, labels = next(s)
        self.assertIn(ord('K')+4, labels)
        self.assertNotIn(ord('J')+4, labels)

    def test_detect_modified_shard(self):
        manifest, shard = fixture(self.base, 'pretrain', [{'text':'abcdef'}])
        s = ShardTextStream(manifest,self.tokenizer,mode='pretrain',context=4)
        before = s.fingerprint
        shard.write_text('{"text":"autre"}\n', encoding='utf-8')
        self.assertNotEqual(before, ShardTextStream(manifest,self.tokenizer,mode='pretrain',context=4).fingerprint)

    def test_checkpoint_weights_optimizer_rng_and_cursor(self):
        manifest, _ = fixture(self.base, 'pretrain', [{'text':'Bonjour '*200}])
        data = ShardTextStream(manifest,self.tokenizer,mode='pretrain',context=16)
        model = Tiny()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        ids, labels = next(data)
        seq = torch.tensor(ids,dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(model(seq[:-1]), seq[1:])
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        output = self.base/'runs'
        weights_before = [p.detach().clone() for p in model.parameters()]
        save_full_checkpoint(output, model, self.tokenizer, opt, data, step=5,
                             engine='pytorch',options={'context':16})
        self.assertEqual(resolve_checkpoint(output).name,'step-00000005')
        other = Tiny()
        state = torch.load(resolve_checkpoint(output)/'model'/'tiny_state.pt', weights_only=True)
        other.load_state_dict(state)
        opt2 = torch.optim.AdamW(other.parameters(), lr=1e-3)
        data2 = ShardTextStream(manifest,self.tokenizer,mode='pretrain',context=16)
        self.assertEqual(load_full_checkpoint(resolve_checkpoint(output),opt2,data2,
                                              engine='pytorch',options={'context':16}),5)
        for a,b in zip(weights_before,other.parameters()):
            self.assertTrue(torch.equal(a,b))
        self.assertEqual(next(data),next(data2))
        self.assertTrue(opt2.state_dict()['state'])
        with self.assertRaises(ValueError):
            load_full_checkpoint(resolve_checkpoint(output),opt2,data2,
                                 engine='pytorch',options={'context':1024})


if __name__ == '__main__':
    unittest.main()
