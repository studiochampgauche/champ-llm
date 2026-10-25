#!/usr/bin/env python3
"""Tools for the champ-llm local training pipeline."""
import importlib.util
import os
# Keep this validation on CPU, as with the vision cycle.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import unittest
from pathlib import Path
for module in ('torch','transformers','tokenizers','safetensors'):
    if importlib.util.find_spec(module) is None:
        raise SystemExit('Missing dependency for the real text test: '+module)
base=Path(__file__).resolve().parent
suite=unittest.TestLoader().discover(str(base/'tests'), pattern='test_real_llama_text.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or result.testsRun != 1:
    raise SystemExit('Real Llama text test incomplete: do not start a long paid training run.')
print('OK: pretraining, conversations, saving and resumption on a real tiny Llama.')
