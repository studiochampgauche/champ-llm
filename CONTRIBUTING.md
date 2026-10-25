# Contributing to champ-llm

Contributions can improve training, data preparation, evaluation, browser
integration and documentation.

## Good places to start

- Improve evaluations for multilingual conversation, code correctness and visual feedback.
- Make dataset validation, training diagnostics or checkpoint handling more reliable.
- Reproduce browser tests on supported operating systems and document issues.
- Contribute small, verified examples with clear provenance.

Discuss substantial architecture or dependency changes in an issue before starting
a large implementation.

## Work locally

Follow [Getting started](GETTING_STARTED.md). Install the appropriate PyTorch build,
the core, vision and corpus dependencies, and the Playwright browsers, then run:

```sh
python -m unittest discover -s tests -v
```

For changes limited to browser access:

```sh
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

See [Validation](docs/VALIDATION.md) for integration checks and browser selection.
Report skipped tests separately from passing tests.

Keep changes focused. In a pull request, explain the problem, the resulting
behavior and how you verified it.

## Code and data conventions

Use English for code identifiers, CLI options, configuration and documentation.
Natural-language examples may be multilingual. Use UTF-8 and platform-neutral paths.

Keep training corpora, model weights, generated screenshots, reports and credentials
out of source changes. Test fixtures should be small and temporary. Discuss the
provenance, license and evaluation split of a proposed public dataset before adding it.

For documentation, describe the current behavior and the steps a reader needs.
Put technical detail in the relevant guide. Keep private setup notes, unpublished
migration history and task-specific progress reports out of public documentation.

## Reporting a problem

Include the smallest reproduction, expected and actual behavior, operating system,
package versions and whether the failure occurs on CPU, CUDA or in the browser.
Remove credentials, private data and personal paths from shared logs.

The project uses the [MIT license](LICENSE). Third-party code and data retain their
applicable terms.
