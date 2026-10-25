# Contributing to champ-llm

Useful contributions make the training pipeline easier to inspect, reproduce or
evaluate. The project is early-stage: there is no trained model release or bundled corpus.

## Good places to start

- Reproduce the browser test on supported Linux or macOS installations and report
  the OS, Python and Playwright versions.
- Improve held-out evaluations for multilingual conversation, code correctness and
  visual feedback.
- Improve reproducibility, dataset validation or training diagnostics.
- Develop small, verified examples of web-development tasks with clear provenance.

Discuss substantial architecture or dependency changes in an issue before
investing in a large implementation.

## Work locally

Follow [Getting started](GETTING_STARTED.md). Use a fresh virtual environment,
install the appropriate PyTorch build and install Chromium, Firefox and WebKit when testing capture. The browser guide also
explains how to include installed Google Chrome.

```sh
python -m unittest discover -s tests -v
```

For changes limited to browser access:

```sh
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

Keep changes focused. Include what changed, why it matters, the command you ran and
the observed result. A test that was skipped is not a verified capability.

## Code and data conventions

Use English names for code, CLI options, configuration and technical documentation.
Natural-language examples may be multilingual. Use UTF-8 files and platform-neutral
paths; never embed a developer's local directory or browser executable.

Keep training corpora, model weights, screenshots, generated reports and credentials
out of ordinary source changes. Test fixtures should be small and temporary.
If proposing a public dataset, discuss its provenance, license and evaluation split
separately before adding it.

A browser-corrected page is not automatically a good training example. Check the
result and its tests before proposing it for a future dataset.

## Reporting a problem

Describe the smallest reproduction, expected and actual behavior, relevant package
versions and whether the failure happens on CPU, CUDA or in the browser. Remove
personal paths, credentials and private training content from logs before sharing.

The source project uses the [MIT license](LICENSE). Third-party code and data must
retain their applicable terms.
