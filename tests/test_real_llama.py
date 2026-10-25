"""Tools for the champ-llm local training pipeline."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
from safetensors.torch import load_file

from capture_website import create_annotation
from prepare_vision import prepare_vision
from train_vision_pytorch import parse_args, train
from vision_checkpoints import checkpoint_path
from analyze_image import analyze
from vision_integrity import sha256_file

HAVE_LLAMA = bool(importlib.util.find_spec('transformers') and importlib.util.find_spec('tokenizers'))


@unittest.skipUnless(HAVE_LLAMA, 'Install transformers and tokenizers for the real Llama test')
class RealTinyLlamaCycleTests(unittest.TestCase):
    def test_train_save_resume_and_analyze_image(self):
        from transformers import LlamaConfig, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

        with tempfile.TemporaryDirectory(prefix='champ-real-llama-') as td:
            root=Path(td)
            model_dir=root/'model'
            model_dir.mkdir()

            
            raw=Tokenizer(models.BPE(unk_token='<unk>'))
            raw.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
            raw.decoder=decoders.ByteLevel()
            examples=['Utilisateur: Analyse cet écran\nAssistant: Fond bleu clair.',
                      'Utilisateur: Trouve les défauts visuals\nAssistant: La typographie manque de contraste.',
                      'header section footer button background contrast responsive',
                      'The page contains a navigation menu and a blue button.']
            raw.train_from_iterator(examples*5, trainer=trainers.BpeTrainer(vocab_size=220, min_frequency=1,
                                special_tokens=['<pad>','<bos>','<eos>','<unk>']))
            tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw, pad_token='<pad>',
                                               bos_token='<bos>', eos_token='<eos>', unk_token='<unk>')
            tokenizer.save_pretrained(model_dir)
            config=LlamaConfig(vocab_size=len(tokenizer),hidden_size=64,intermediate_size=128,
                              num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,
                              max_position_embeddings=512,hidden_act='silu',
                              pad_token_id=tokenizer.pad_token_id,
                              bos_token_id=tokenizer.bos_token_id,eos_token_id=tokenizer.eos_token_id,
                              tie_word_embeddings=False)
            torch.manual_seed(123)
            original=LlamaForCausalLM(config)
            self.assertGreater(sum(p.numel() for p in original.parameters()),0)
            original.save_pretrained(model_dir,safe_serialization=True)
            reloaded=AutoModelForCausalLM.from_pretrained(str(model_dir),local_files_only=True)
            self.assertIsInstance(reloaded,LlamaForCausalLM)
            del original,reloaded

            folder=root/'training_data'/'vision'/'test'
            folder.mkdir(parents=True)
            im=folder/'ecran.png'
            Image.new('RGB',(180,130),'lightblue').save(im)
            sidecar=create_annotation(im,project='test',width=375)
            ann=json.loads(sidecar.read_text(encoding='utf-8'))
            ann['description']='Capture d’une interface avec fond bleu.'
            ann['critique']='La hiérarchie visual est insuffisante.'
            sidecar.write_text(json.dumps(ann,ensure_ascii=False),encoding='utf-8')
            report=prepare_vision(root/'training_data',root/'outputs',validation=0,root=root)
            self.assertEqual(report['annotated_images'],1)
            manifest=root/'outputs'/'vision'/'vision_manifest.json'

            def opts(out,steps,*flags):
                return parse_args(['--model',str(model_dir),'--manifest',str(manifest),
                                   '--output',str(out),'--steps',str(steps),'--image-size','32',
                                   '--max-tiles','2','--pool-grid','1','--vision-dim','16',
                                   '--vision-layers','1','--max-prompt-tokens','80',
                                   '--max-answer-tokens','128','--save-every','1',*flags])

            from importlib import import_module
            trainer=import_module('train_vision_pytorch')
            with patch.object(trainer,'ROOT',root):
                train(opts(root/'continuous',2))
                train(opts(root/'resumed',1))
                train(opts(root/'resumed',1,'--resume'))

            reference=checkpoint_path(root/'continuous',resume=True)
            resumed=checkpoint_path(root/'resumed',resume=True)
            self.assertEqual(json.loads((resumed/'metadata.json').read_text())['step'],2)
            wa=load_file(str(reference/'vision.safetensors'))
            wb=load_file(str(resumed/'vision.safetensors'))
            for name in wa:
                self.assertTrue(torch.equal(wa[name],wb[name]),name)
            self.assertTrue(any(torch.any(v != 0) for v in wb.values()))
            reloaded=AutoModelForCausalLM.from_pretrained(str(model_dir),local_files_only=True)
            tok_reloaded=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
            answer=analyze(resumed,im,'Que vois-tu ?',reloaded,tok_reloaded,max_new_tokens=5)
            self.assertIsInstance(answer,str)
            self.assertEqual(64,reloaded.config.hidden_size)
            print('REAL LLAMA CYCLE VALIDATED: training, checkpoint, exact resumption and image analysis.')
            print('Raw response (random weights; meaningful output is not guaranteed):',repr(answer))


if __name__ == '__main__':
    unittest.main()
