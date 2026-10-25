# Vision, browser screenshots and feedback

## Choose a browser

Install the Playwright-managed engines into your current Python environment:

```sh
python -m pip install -r requirements-vision.txt
python -m playwright install chromium firefox webkit
```

| CLI choice | Browser actually used |
| --- | --- |
| `--browser chromium` | Playwright's Chromium; the default when no browser is specified |
| `--browser chrome` | Installed Google Chrome, launched through Playwright's `chrome` channel |
| `--browser firefox` | Playwright's Firefox build |
| `--browser webkit` | Playwright's WebKit build |
| `--browser safari` | Explicit alias for `webkit`; not the installed Safari application |

Google Chrome must already be installed on the machine. The application does not
install, replace or update your personal browser. Firefox uses Playwright's build,
which includes the automation patches required by Playwright.

WebKit is the engine used by Safari, but the Playwright build is not a guarantee
of identical behavior in a released Safari version. WebKit on macOS gives the
closest platform behavior. See [Playwright's browser documentation](https://playwright.dev/python/docs/browsers).
The chosen browser is never silently replaced with a different engine on failure.

## Capture one or several browsers

```sh
python capture_website.py http://localhost:3000 --project my-site --browser firefox
python capture_website.py http://localhost:3000 --project my-site --browser chrome --browser firefox --browser safari
```

Repeat `--browser` to inspect several browsers in one command. Duplicate engines
are inspected once; specifying both `safari` and `webkit` selects WebKit once.
Each browser uses three viewports: mobile 375, tablet 768 and desktop 1440.
The same explicit click/fill/press scenarios run in every browser and viewport.

Whole-page and HTML-section screenshots are saved under
`training_data/vision/<project>/<browser>/`. The browser directory is `webkit`
for both WebKit and the Safari alias. Screenshots keep their content-hashed names,
and each `.vision.json` annotation records the image SHA-256 plus
`browser_metadata` (browser, engine, channel and version). Reviewed annotations
remain bound to their original screenshot.

Technical reports are saved as `outputs/audits_browser/<project>-<browser>.json`.
This keeps captures and reports from different browsers separate, even when their
pixels happen to match. Existing captures from older releases are not moved.
Reports detect technical issues such as JavaScript errors, overflow and missing
alt text; they do not intrinsically judge visual quality. Use `--allow-external`
only for sites you are authorized to inspect.

## Use the same browsers in the development loop

```sh
python coding_agent.py --project ../my-site --url http://localhost:3000 --goal "Fix responsive layout" --browser chrome --browser firefox --browser safari
```

The selected browsers are checked before and after approved edits. Browser names
are attached to issues and recorded in the run report. With a trained vision
backend, up to two screenshots per browser are reviewed, prioritizing mobile and
desktop. Without vision, the agent still receives the technical browser diagnostics.
See [the coding-agent guide](DIRECT_AGENT.md) for model setup and `--apply`.

## Run the real browser test on Windows, Linux or macOS

Use the same Python environment for installation and testing:

```sh
python -m playwright install chromium firefox webkit
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

By default this tests Chromium, Firefox and WebKit. To include installed Google
Chrome, explicitly select all four browsers for the test run:

Windows / PowerShell:

```powershell
$env:CHAMP_LLM_TEST_BROWSERS = "chromium,chrome,firefox,safari"
python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

Linux / macOS:

```sh
CHAMP_LLM_TEST_BROWSERS=chromium,chrome,firefox,safari python -m unittest discover -s tests -p test_real_browser_agent.py -v
```

The same variable applies to the complete test suite. It can select an individual
browser for a focused run; the result then validates only that selected browser.
Missing selected browsers, launch failures or blocked localhost access fail the
test instead of being skipped. A failed engine is not substituted with Chromium.

Playwright selects the engine builds for the current supported operating system.
On supported Linux distributions, install missing system libraries with
`python -m playwright install --with-deps chromium firefox webkit` (system package
installation may require administrator privileges). Consult the
[official system requirements](https://playwright.dev/python/docs/intro#system-requirements).
A passing run validates the OS where it ran, not untested operating systems.

For a custom compatible Chromium executable, optionally set `CHAMP_LLM_CHROMIUM`.
The earlier `CHAMP_GAUCHE_LLM_CHROMIUM` and `MONCERVEAU_CHROMIUM` names remain
fallbacks, in that order. This override applies only to `--browser chromium`;
it does not redirect Chrome, Firefox or WebKit. If using a custom
`PLAYWRIGHT_BROWSERS_PATH`, use the same value for installation and execution.

The real browser test starts a temporary local HTTP server, fills a field, presses
Enter, clicks a button, detects overflow, applies a predefined correction and
checks the result at all three viewports in each selected browser. It also checks
section captures, separate browser artifacts, browser metadata and screenshot
versions. Only model responses are simulated. All site files and captures are
temporary; they never enter the project's training corpus.

On Windows, keep project and temporary paths short when long-path support is
unavailable. Deep paths can exceed the Windows limit for screenshot sidecars.

## Annotate screenshots

Edit the sidecar JSON, not the screenshot, after reviewing the screenshot. English metadata keys include `project`, `phase`, `question`, `description`, `critique`, `improvements`, `correction`, and `code_files`. Text values may be written in French. `code_files` entries are relative to `training_data/`. The SHA-256 binding prevents attaching an old critique silently to a changed image.

Example:

```json
{
  "project":"my-site",
  "phase":"before-fix",
  "question":"Qu'est-ce qui ne va pas avec cette interface ?",
  "description":"Page d'accueil avec menu et cartes",
  "critique":"Le contraste du titre est insuffisant",
  "improvements":"Augmenter le contraste et corriger les espaces",
  "correction":"Corriger les styles CSS du titre",
  "code_files":["programming/react/Button.jsx"]
}
```

The actual generated sidecar will also include `image_sha256`; **do not delete or invent this field**. For legacy unhashed annotations, review the image first, then use `bind_annotation.py --image <path> --confirm`.

## Prepare the vision dataset

```powershell
python prepare_vision.py
# Or prepare text + vision together:
python prepare_data.py
```

`outputs/vision/vision_manifest.json` points to the visual shards. For GPU vision-training smoke tests:

```bash
python train_vision_pytorch.py --steps 1 --image-size 224 --max-tiles 9
```

Train/restore the **same** model and vision checkpoint for inference. The vision module is a prototype with randomly initialized weights; images become meaningful to the model only after significant paired image/text/code training. Whole-page images are tiled, but this does not guarantee all visual detail is preserved. Run real-model integration tests before long GPU sessions.
