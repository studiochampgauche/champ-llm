"""Tools for the champ-llm local training pipeline."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

HAVE_HF = all(importlib.util.find_spec(x) is not None for x in ('transformers','tokenizers','safetensors'))


@unittest.skipUnless(HAVE_HF, 'Install transformers and tokenizers for the real model test')
class RealLocalBackendTests(unittest.TestCase):
    def test_load_generate_and_process_image_without_ollama(self):
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
        from safetensors.torch import save_file
        from local_backend import ChampBackend
        from vision_model import VisionEncoder

        with tempfile.TemporaryDirectory(prefix='champ-direct-') as temporary:
            root = Path(temporary)
            model_dir = root/'model'
            model_dir.mkdir()
            bpe = Tokenizer(models.BPE(unk_token='<unk>'))
            bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            bpe.decoder = decoders.ByteLevel()
            specials = ['<pad>', '<bos>', '<eos>', '<unk>', '<|system|>', '<|user|>', '<|assistant|>', '<|tool|>', '<|end|>']
            bpe.train_from_iterator([
                '<|user|>\nFais une petite interface en React.\n<|end|>\n<|assistant|>\nVoici un composant React.',
                '<|user|>\nAnalyse cette image.\n<|end|>\n<|assistant|>\nIl faut améliorer la typographie.',
            ] * 10, trainer=trainers.BpeTrainer(
                vocab_size=256, min_frequency=1, special_tokens=specials,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
            tok = PreTrainedTokenizerFast(tokenizer_object=bpe, pad_token='<pad>',
                bos_token='<bos>', eos_token='<eos>', unk_token='<unk>',
                additional_special_tokens=specials[4:])
            tok.save_pretrained(model_dir)
            llama = LlamaForCausalLM(LlamaConfig(
                vocab_size=len(tok), hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=1024, bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id))
            llama.save_pretrained(model_dir, safe_serialization=True)
            del llama
            ckpt = root/'visual'
            ckpt.mkdir()
            bridge = VisionEncoder(64, image_size=32, patch_size=16, vision_dim=16,
                                    vision_layers=1, max_tiles=2, pool_grid=1)
            save_file({k: v.detach().contiguous() for k,v in bridge.state_dict().items()},
                      str(ckpt/'vision.safetensors'))
            (ckpt/'vision_config.json').write_text(json.dumps(bridge.vision_config))
            (ckpt/'metadata.json').write_text(json.dumps({'mode':'vision-only'}))
            img=root/'site.png'
            Image.new('RGB',(60,150),'white').save(img)
            backend = ChampBackend(model_dir, vision=ckpt, device='cpu',
                                         max_generation=4, max_vision_generation=3)
            self.assertTrue(backend.has_vision)
            self.assertIsInstance(backend.generate_code('Bonjour'), str)
            self.assertIsInstance(backend.critique_images([img]), str)

if __name__ == '__main__':
    unittest.main()
