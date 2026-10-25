# Datasets and languages

Build your corpus around the languages and domains you want your model to learn.
Text, code and conversations can be multilingual; use the formats below to prepare them.

## Raw text and source code

Place UTF-8 files under `training_data/`. Nested folders help organize languages,
topics and projects. You can also use a root-level `training_data.txt`; both inputs
are read during preparation.

For example:

```text
training_data/
├── french/greetings.txt
├── english/conversation.txt
├── spanish/conversation.txt
├── programming/php/index.php
├── programming/react/Button.jsx
├── programming/scss/cards.scss
└── conversations/questions.jsonl
```

`prepare_data.py` scans these inputs, filters known generated or sensitive files,
and writes bounded shards. Source-file records include path and language metadata
alongside their content. See [Getting started](GETTING_STARTED.md) for preparation
and tokenizer commands.

For larger French/English collections assembled from multiple sources, see
[Corpus preparation](CORPUS_PIPELINE.md).

## Conversation format

Use JSONL: one JSON object per line, with a `messages` array. Each message has a
`role` and `content`. Supported roles are `system`, `user`, `assistant` and `tool`.
Content can use any language your model is being trained to handle.

```json
{"messages":[{"role":"user","content":"Explique les variables PHP."},{"role":"assistant","content":"En PHP, les variables commencent par $."}]}
```

Conversation training learns from assistant responses. Set `"learn": false` on
an assistant message to keep it as context while excluding that response from
the training loss.

See [Conversations and checkpoints](CONVERSATIONS_CHECKPOINTS_AGENT.md) for training
and resumption.

## Images and code

Use `capture_website.py` to generate PNG screenshots and matching `.vision.json`
annotation files. Review each screenshot and fill in `description`, `critique`,
`improvements`, `correction` and `code_files`.

Image paths and referenced code files must resolve inside `training_data/`.
Each annotation includes the screenshot's hash so it stays associated with the
correct image. See [Vision and browser](VISION_AND_BROWSER.md) for the complete
annotation format and preparation commands.

## Dataset quality

Keep evaluation examples separate from training data, including closely related
documents and conversations. Review provenance, applicable licenses, factual
quality and code correctness before adding material. Automated secret filters
assist review but cannot detect every credential or piece of personal information.

Measure progress on held-out tasks as well as training loss. Preserve the prepared
manifest and shards for any run you may need to resume; its saved data position
depends on that dataset snapshot.

Keep the tokenizer fixed once model training starts. New datasets can be prepared
with the existing tokenizer. Training data and generated artifacts are ignored by
Git by default.
