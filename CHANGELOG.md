# Release notes

<!-- do not remove -->

## 0.0.10

### Breaking Changes

- Rename the `completion_model` config and `%ipyai` setting to `suggest_model`; existing `config.json` files must update that key.

## 0.0.9

### New Features

- Add `codex-api` and make it the default backend ([#9](https://github.com/AnswerDotAI/ipyai/issues/9))
- Switch from claude sdk to claude -p ([#8](https://github.com/AnswerDotAI/ipyai/issues/8))
- Refactor backends to share BaseBackend/ConversationSeed/CommonStreamFormatter via new `backend_common` module ([#4](https://github.com/AnswerDotAI/ipyai/issues/4))
- Add MCP tool prefix stripping, tool start/complete display for codex client, and trailing blank-line fix in compact tool summaries ([#3](https://github.com/AnswerDotAI/ipyai/issues/3))
- Add cancellation support, fix session reinit, and capture Codex stream sample ([#2](https://github.com/AnswerDotAI/ipyai/issues/2))
- Consolidate CLI to single ipyai entry point with per-backend model config and config migration ([#1](https://github.com/AnswerDotAI/ipyai/issues/1))


## 0.0.8

- Multi-backend with codex, lisette, and claude agent sdk


## 0.0.1

- init commit

