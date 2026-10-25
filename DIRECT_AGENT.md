# Direct local coding agent

Use a trained local model to inspect a running website, propose code changes and
check them with browser diagnostics and your project's tests. A compatible trained
vision checkpoint can add feedback on screenshots. You can also use an Ollama backend.

## Preview changes

Start your development server, then point the agent at the project and model:

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Improve the mobile layout" --backend local --local-model model
```

Write the goal in a language your model understands. The command previews proposed
changes; add `--apply` when you want the agent to write them.

## Choose browsers

The default is Chromium. Repeat `--browser` to inspect several browsers:

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Improve the mobile layout" --backend local --local-model model --browser chrome --browser firefox --browser safari
```

Chrome uses installed Google Chrome. Firefox uses Playwright's Firefox build.
`safari` selects Playwright WebKit, rather than the Safari application.
Install the selected browsers using the [browser setup guide](VISION_AND_BROWSER.md).

## Apply changes and run tests

Work in a separate Git branch or worktree so changes are easy to inspect and undo.
After reviewing a preview, allow file changes and supply your project's test command:

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Fix the mobile menu" --backend local --local-model model --tests "npm run build" --apply
```

The agent checks the selected browsers before and after edits. Use `--interactions`
with a scenario file to exercise clicks, field entry and key presses; see
[the interaction example](INTERACTION_SCENARIOS_EXAMPLE.json).

## Model requirements

Use trained weights for coding tasks. Visual feedback additionally requires a
trained, compatible vision module. The framework is experimental: review the
resulting code and evaluate it with your own tests.

The agent does not update model weights during use. Save reviewed corrections
for a later training cycle if you want the model to learn from them.
