# Release notes

<!-- do not remove -->

## 0.1.0

### New Features

- Move to jupygate kernels and file-based sessions, dropping the sqlite store, with kernel-side sig help and refs ([#26](https://github.com/AnswerDotAI/ipyai/issues/26))
- Refactor `claude_code` transport registration into `_split_vendor` helper ([#22](https://github.com/AnswerDotAI/ipyai/pull/22)), thanks to [@ncoop57](https://github.com/ncoop57)
- Use the message model from aidialog rather than fastllm ([#20](https://github.com/AnswerDotAI/ipyai/pull/20)), thanks to [@jph00](https://github.com/jph00)
- Replace argparse with fastcore `@call_parse`, drop -b backend sugar in favor of per-key CLI flags, make plain launch start fresh with explicit -r resume, bump default models, and fix mdhtml .data→.text ([#19](https://github.com/AnswerDotAI/ipyai/issues/19))
- lisette -> fastllm [3/3] ([#18](https://github.com/AnswerDotAI/ipyai/pull/18)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)
- Use minimal tools (pyrun,bash w pyskills) & append instead of replace user's sysp.txt [2/3] ([#17](https://github.com/AnswerDotAI/ipyai/pull/17)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)
- Replace CodexChat with AsyncChat + chatgpt/ provider prefix in CodexAPIBackend ([#14](https://github.com/AnswerDotAI/ipyai/issues/14))
- Add F2 open-in-editor binding, history ghost text docs, and stream rendering fixes ([#13](https://github.com/AnswerDotAI/ipyai/issues/13))
- Add IPython history autosuggest provider ([#12](https://github.com/AnswerDotAI/ipyai/issues/12))
- Preserve full tool results through bridge/formatter; add --existing resume hint, kernel debug flag, longer tool timeout ([#11](https://github.com/AnswerDotAI/ipyai/issues/11))
- Switch to jupyter console based ([#10](https://github.com/AnswerDotAI/ipyai/issues/10))
- Add ipyclaude and ipycodex CLI entry points with preset backends ([#7](https://github.com/AnswerDotAI/ipyai/pull/7)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)

### Bugs Squashed

- Fix for tests on main [1/3] ([#16](https://github.com/AnswerDotAI/ipyai/pull/16)), thanks to [@RensDimmendaal](https://github.com/RensDimmendaal)
- Serializing tool calls ([#15](https://github.com/AnswerDotAI/ipyai/pull/15)), thanks to [@kafkasl](https://github.com/kafkasl)


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
