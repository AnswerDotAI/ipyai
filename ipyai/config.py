"Config for the ipyaing app: same files as classic ipyai (XDG config.json + sysp.txt), IPython-free."
import json, os
from pathlib import Path
from fastcore.xdg import xdg_config_home

DEFAULT_MODEL = 'codex/gpt-5.6-terra'
DEFAULT_SUGGEST_MODEL = 'codex/gpt-5.6-luna'
DEFAULT_THINK = 'm'
DEFAULT_CODE_THEME = 'auto'   # auto: pick ansi_dark/ansi_light from the terminal background (OSC 11)
DEFAULT_PROMPT_MODE = False
DEFAULT_PROMPT_STYLE = ''     # Rich style string for prompt text, e.g. 'bold'; '' leaves it plain
DEFAULT_PAD_TRANSCRIPT = False  # blank row before each input block, spacing the transcript by turns
CONFIG_DIR = xdg_config_home()/'ipyai'
CONFIG_PATH = CONFIG_DIR/'config.json'
SYSP_PATH = CONFIG_DIR/'sysp.txt'

DEFAULT_SP = """You are an AI assistant running inside terminal IPython through ipyai.

The user's message may include, before the request itself:
- `<code>` blocks: recently executed Python cells, with their outputs
- `<markdown>` blocks: user-authored notes (ctx, not executable code)
- `<variable>` blocks: live interpreter values
- `<shell>` blocks: shell command output

Use tools when they materially improve correctness:
- use live Python tooling such as `py` when interpreter state matters
- use available shell/file tools for repository work
- use web tools when fresh web information matters

Respond concisely and practically. Markdown is rendered in a terminal with Rich."""
SUGGEST_SP = 'You are a code suggestion engine for IPython. Return only the suggestion text to insert at the cursor.'

def _default_config():
    return dict(model=DEFAULT_MODEL, suggest_model=DEFAULT_SUGGEST_MODEL, think=DEFAULT_THINK,
                code_theme=DEFAULT_CODE_THEME, prompt_mode=DEFAULT_PROMPT_MODE,
                prompt_style=DEFAULT_PROMPT_STYLE, pad_transcript=DEFAULT_PAD_TRANSCRIPT)

def load_config(path=None):
    "Effective config dict: file over defaults; models are flat vendor-prefixed strings like 'codex/gpt-5.4'."
    path = Path(path or CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists(): data = json.loads(path.read_text())
    else: path.write_text(json.dumps(_default_config(), indent=2) + '\n')
    cfg = _default_config() | {k: v for k, v in data.items() if k in _default_config()}
    cfg['model'] = str(cfg['model'] or os.environ.get('IPYAI_MODEL', '') or DEFAULT_MODEL).strip()
    cfg['suggest_model'] = str(cfg['suggest_model'] or DEFAULT_SUGGEST_MODEL).strip()
    cfg['think'] = str(cfg['think'] or DEFAULT_THINK).strip().lower()
    cfg['code_theme'] = str(cfg['code_theme']).strip() or DEFAULT_CODE_THEME
    cfg['prompt_mode'] = bool(cfg['prompt_mode'])
    cfg['prompt_style'] = str(cfg['prompt_style']).strip()
    cfg['pad_transcript'] = bool(cfg['pad_transcript'])
    return cfg

def load_sysp(path=None):
    path = Path(path or SYSP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_text(DEFAULT_SP)
    return path.read_text()
