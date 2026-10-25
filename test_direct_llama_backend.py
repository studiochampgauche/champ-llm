#!/usr/bin/env python3
"""Tools for the champ-llm local training pipeline."""
import importlib.util
import sys
import unittest
from pathlib import Path
for mod in ('torch', 'transformers', 'tokenizers', 'safetensors', 'PIL'):
    if importlib.util.find_spec(mod) is None:
        raise SystemExit(f'Missing dependency: {mod}. Install requirements.txt and PyTorch.')
root=Path(__file__).resolve().parent
suite=unittest.TestLoader().discover(str(root/'tests'),pattern='test_local_backend_real_llama.py')
result=unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped or not result.wasSuccessful() or result.testsRun != 1:
    raise SystemExit('Real tiny Llama backend NOT VALIDATED. Do not start a long GPU training run.')
print('OK: direct HF Llama loading, code generation and vision adapter without Ollama.')
