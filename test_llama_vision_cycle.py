#!/usr/bin/env python3
"""Tools for the champ-llm local training pipeline."""
import importlib.util
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
import unittest
from pathlib import Path

for name in ('torch','transformers','tokenizers','safetensors','PIL'):
    if importlib.util.find_spec(name) is None:
        raise SystemExit(f'MISSING {name}. Install dependencies before running this real model test.')
base=Path(__file__).resolve().parent
suite=unittest.TestLoader().discover(str(base/'tests'),pattern='test_real_llama.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if not result.wasSuccessful() or result.skipped or result.testsRun != 1:
    raise SystemExit('FAILED / TEST NOT EXECUTED: do not start a long GPU training run.')
print('OK: the complete LlamaForCausalLM cycle succeeded on this machine.')
