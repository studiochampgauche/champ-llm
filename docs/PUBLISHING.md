# Publishing this source project

The source tree is prepared for sharing. It contains no trained weights or corpus.
The MIT license is included. Local environments and generated artifacts are ignored.

## Suggested GitHub presentation

**Repository name**

`champ-llm`

**About**

> An experimental toolkit by Studio Champ Gauche to create and train language models from scratch, with selectable sizes. Exploring multilingual conversation, web development and visual feedback.

**Topics**

`llm` · `pytorch` · `language-model` · `multimodal` · `web-development` ·
`react` · `nodejs` · `gsap` · `playwright` · `machine-learning`

The README includes a local SVG banner, an architecture diagram, entry points,
an honest implementation status and a path for contributors. It does not depend
on a specific GitHub username, repository URL or build badge.

## Before the first push

Use GitHub Desktop or Git to review the files selected for the first commit.
The public selection should contain code, tests, guides, assets and LICENSE.
The raw training inputs, local logs, virtual environment and model artifacts must
remain outside that selection.

The public validation summary is in [VALIDATION.md](VALIDATION.md). Detailed logs
are local artifacts and should not be attached to issues without review.

## Command-line publication

Create an empty repository on GitHub, without generating an extra README or license.
Then run the following from this project directory, replacing the example account
in the remote URL with your own:

```sh
git init -b main
git add .
git status --short
git diff --cached --stat
git commit -m "Publish champ-llm"
git remote add origin https://github.com/YOUR-ACCOUNT/champ-llm.git
git push -u origin main
```

Review the staged file list before committing. Ignore rules do not remove files
that have already been tracked in an existing repository.

If using a prepared source ZIP, extract it first and publish its contents as the
repository files. Uploading only the ZIP does not expose the README and source tree
as a normal GitHub repository.

## What is deliberately local?

- `training_data/` and optional `training_data.txt`
- `model/`, `models/`, `training_runs/` and model/checkpoint binaries
- `outputs/` except its short directory README
- Virtual environments, caches, logs, archives and machine configuration

Git does not preserve empty folders. The setup guide shows how to create the local
data directory. The source license does not automatically license future datasets
or model weights.
