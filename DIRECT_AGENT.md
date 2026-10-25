# Direct local coding agent

The agent can call your **trained** `champ-llm` weights through `local_backend.py` instead of Ollama. A trained, compatible vision checkpoint can provide additional visual critiques. An untrained model will not produce reliable code or design feedback.

## Preview without writing files

Start your development server separately (for example at `http://localhost:3000`), then:

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Améliore ce site sans casser le responsive" --backend local --local-model model
```

The command-line flags, JSON keys, and programming identifiers are English. **Your goal can still be in French.** The agent prompt asks the model to respond in the goal's language and to use English code identifiers.

## Choose browsers

The agent defaults to Chromium. Select a different browser or repeat `--browser`
to compare several engines before and after edits:

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Corrige le responsive" --backend local --local-model model --browser chrome --browser firefox --browser safari
```

Chrome uses your installed Google Chrome. Firefox uses Playwright's Firefox build.
`safari` is an explicit alias for Playwright WebKit; it does not launch Safari.
Missing selected browsers cause an error, with no fallback to another engine.
Issues and visual critiques retain their browser identity. See
[installation and browser tests](VISION_AND_BROWSER.md).

## Apply authorized edits

After checking its proposals, explicitly opt in to file writes:

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Corrige le menu mobile" --backend local --local-model model --tests "npm run build" --apply
```

Use a Git worktree, disposable development environment and backups. The AI cannot safely be granted arbitrary shell or production-server access. The browser supports local URLs by default and explicit click/fill/press interaction scenarios in `INTERACTION_SCENARIOS_EXAMPLE.json`. An optional Ollama backend is provided for testing the agent before training your own model.

This is an **experimental agent framework**, not a production autonomous developer. Model quality, design judgment, code correctness and security must be validated separately. The current agent only uses screenshots with a trained vision module; it cannot learn from its runtime failures unless those corrections are later recorded and fed back into training.
