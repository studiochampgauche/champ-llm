# Datasets and languages

Code names, metadata, tooling, files and folders are **English**. This does **not** limit what the model can learn: training text and user messages may be French, English, Spanish, or other languages. Do not translate French conversations into English just to follow the repository naming convention.

## Raw text and source code

`prepare_data.py` recursively scans `training_data/`, skips sensitive/known generated files, and streams UTF-8 text into bounded shards. It supports nested folders, common web languages and custom UTF-8 text extensions. The corpus starts empty; the following tree is a suggested layout, not bundled training data:

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

File examples are rendered with English labels such as `Path:`, `Language:`, `Extension:`, `Section:`, `Part:`, and `Content:` so the model can learn file context as well as code syntax. A filename is metadata, not magical knowledge; the model must learn how to use it.

## Conversation SFT format

One JSON object per line, with English **keys** and any language in **content**:

```json
{"messages":[{"role":"user","content":"Explique les variables PHP."},{"role":"assistant","content":"En PHP, les variables commencent par $."}]}
```

Optional `"learn": false` on an assistant message masks that response from supervised training, while keeping it in the conversation context. The old `"apprendre"` key is not the new schema: update your data before using it. Supported roles include `system`, `user`, `assistant`, `tool`.

## Images and code

Use `capture_website.py` to generate versioned PNG screenshots and `.vision.json` sidecars. Each annotation is bound by SHA-256 to its exact screenshot. Fill in `description`, `critique`, `improvements`, `correction`, and `code_files` before training. Training image paths must resolve inside `training_data/`. The visual dataset is separate from text-only pretraining.

## Safety, evaluation and reproducibility

- Review any code you ingest; remove secrets, personal information, and unlicensed/proprietary training material.
- The preparer scans for common secrets, but **cannot guarantee** that secrets are excluded.
- Train/validation partitioning is by document or conversation; keep a truly held-out evaluation set for long-term comparisons.
- The shard manifest represents one dataset snapshot. A checkpoint's saved shard cursor is only resumable if the same dataset snapshot is retained.
- Do not interpret decreasing training loss as proof the model can reason, code securely, or evaluate design aesthetics.
