# Vision and browser

Capture websites in several browsers, inspect responsive behavior, and prepare
annotated screenshots for visual training.

## Install browsers

```sh
python -m pip install -r requirements-vision.txt
python -m playwright install chromium firefox webkit
```

On supported Linux distributions, install missing system libraries with:

```sh
python -m playwright install --with-deps chromium firefox webkit
```

Playwright provides browser builds for supported Windows, Linux and macOS versions.
See its [system requirements](https://playwright.dev/python/docs/intro#system-requirements)
for supported platforms.

## Choose a browser

| Option | Browser used |
| --- | --- |
| `--browser chromium` | Playwright Chromium; the default |
| `--browser chrome` | Installed Google Chrome |
| `--browser firefox` | Playwright Firefox |
| `--browser webkit` | Playwright WebKit |
| `--browser safari` | Alias for Playwright WebKit |

Install Google Chrome separately to use `--browser chrome`. Firefox uses
Playwright's build. The Safari option uses WebKit, not the Safari application;
test native Safari separately when you need to verify a specific Safari version.
See [Playwright's browser documentation](https://playwright.dev/python/docs/browsers).

## Capture a site

Start the site's development server, then run:

```sh
python capture_website.py http://localhost:3000 --project my-site --browser firefox
```

Repeat `--browser` to capture several browsers:

```sh
python capture_website.py http://localhost:3000 --project my-site --browser chrome --browser firefox --browser safari
```

Each browser is checked at mobile, tablet and desktop widths: 375, 768 and 1440
pixels. Use `--interactions` with [a scenario file](INTERACTION_SCENARIOS_EXAMPLE.json)
to exercise clicks, field entry and key presses.

Screenshots and `.vision.json` annotations are saved under
`training_data/vision/<project>/<browser>/`. Reports are saved under
`outputs/audits_browser/<project>-<browser>.json`. Both retain the browser identity;
the Safari alias uses the `webkit` directory.

Reports detect issues such as JavaScript errors, horizontal overflow and missing
alt text. Use `--allow-external` for an external site you are authorized to inspect.

## Use the coding agent

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Fix the responsive layout" --browser chrome --browser firefox --browser safari
```

The agent uses browser diagnostics to propose changes. A trained vision backend
can also review screenshots. See [the coding-agent guide](DIRECT_AGENT.md) for
model selection, project tests and applying changes.

## Check browser access

```sh
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

The test uses a temporary local site and real Chromium, Firefox and WebKit
instances. Model responses are predefined. To include installed Google Chrome:

Windows / PowerShell:

```powershell
$env:CHAMP_LLM_TEST_BROWSERS = "chromium,chrome,firefox,safari"
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

Linux / macOS:

```sh
CHAMP_LLM_TEST_BROWSERS=chromium,chrome,firefox,safari python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

Selected browsers must be installed and able to launch. Missing browsers cause
a test failure. Use the same Python environment and `PLAYWRIGHT_BROWSERS_PATH`
for browser installation and execution.

For a custom compatible Chromium executable, set `CHAMP_LLM_CHROMIUM`.
On Windows, keep project and temporary paths short if long-path support is unavailable.

## Annotate screenshots

Review each screenshot and edit its matching `.vision.json` file. Text values
can use the languages in your curriculum. For example:

```json
{
  "project": "my-site",
  "phase": "before-fix",
  "question": "What should be improved in this interface?",
  "description": "Homepage with navigation and cards",
  "critique": "The heading has insufficient contrast",
  "improvements": "Increase contrast and adjust spacing",
  "correction": "Update the heading's CSS styles",
  "code_files": ["programming/react/Button.jsx"]
}
```

Keep the generated `image_sha256` field; it binds the annotation to its screenshot.
Paths in `code_files` are relative to `training_data/`. To bind an existing
annotation that has no hash, review the image and run
`python bind_annotation.py --image <path> --confirm`.

## Prepare and train on images

```sh
python prepare_vision.py
```

This creates `outputs/vision/vision_manifest.json` and its referenced shards.
`prepare_data.py` can also prepare text and vision together.

With the training dependencies and a compatible GPU installed, check one step:

```sh
python train_vision_pytorch.py --steps 1 --image-size 224 --max-tiles 9
```

Vision training uses additional steps, including with `--resume`. Use matching
language-model and vision checkpoints for inference.

The vision encoder starts with random weights and requires paired image/text/code
training. It processes still images, with tiling for larger captures; visual quality
and animation understanding require separate evaluation.
