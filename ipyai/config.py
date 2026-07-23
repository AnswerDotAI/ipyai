"Config for the ipyaing app: same files as classic ipyai (XDG config.json + sysp.txt), IPython-free."
import json, os
from pathlib import Path
from fastcore.xdg import xdg_config_home
from .backends import backend_spec, normalize_backend_name, DEFAULT_BACKEND, BACKENDS

DEFAULT_THINK = 'm'
DEFAULT_CODE_THEME = 'auto'   # auto: pick ansi_dark/ansi_light from the terminal background (OSC 11)
DEFAULT_PROMPT_MODE = False
CONFIG_DIR = xdg_config_home()/'ipyai'
CONFIG_PATH = CONFIG_DIR/'config.json'
SYSP_PATH = CONFIG_DIR/'sysp.txt'

DEFAULT_SYSTEM_PROMPT = """You are an AI assistant running inside terminal IPython through ipyai.

The user may give you:
- a `<context>` block containing recent executed Python code, outputs, and notes
- a `<user-request>` block containing the actual request
- `<variable>` blocks containing live interpreter values
- `<shell>` blocks containing shell command output

Treat `<note>` blocks as user-authored context, not executable code.

Use tools when they materially improve correctness:
- use live Python tooling such as `py` when interpreter state matters
- use available shell/file tools for repository work
- use web tools when fresh web context matters

Respond concisely and practically. Markdown is rendered in a terminal with Rich."""
COMPLETION_SP = 'You are a code completion engine for IPython. Return only the completion text to insert at the cursor.'

def _default_config():
    models = {name: dict(model=spec.default_model, completion_model=spec.default_completion_model, think=DEFAULT_THINK)
              for name, spec in BACKENDS.items()}
    return dict(backend=DEFAULT_BACKEND, models=models, code_theme=DEFAULT_CODE_THEME, prompt_mode=DEFAULT_PROMPT_MODE)

def load_config(path=None, backend_name=None):
    "Effective config dict: file over defaults, flattened for the chosen backend (`_backend_name`, `model`, ...)."
    path = Path(path or CONFIG_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _default_config()
    if path.exists():
        data = json.loads(path.read_text())
        for bk, vals in (data.get('models') or {}).items():
            if bk in cfg['models']: cfg['models'][bk].update(vals)
        for k in ('backend', 'code_theme', 'prompt_mode'):
            if k in data: cfg[k] = data[k]
    else: path.write_text(json.dumps(cfg, indent=2) + '\n')
    name = normalize_backend_name(backend_name or cfg['backend'])
    spec = backend_spec(name)
    mcfg = cfg['models'].get(name, {})
    cfg['_backend_name'] = name
    cfg['model'] = str(mcfg.get('model', '') or os.environ.get('IPYAI_MODEL', '') or spec.default_model).strip()
    cfg['completion_model'] = str(mcfg.get('completion_model', '') or spec.default_completion_model).strip()
    cfg['think'] = str(mcfg.get('think') or DEFAULT_THINK).strip().lower()
    cfg['code_theme'] = str(cfg['code_theme']).strip() or DEFAULT_CODE_THEME
    cfg['prompt_mode'] = bool(cfg['prompt_mode'])
    return cfg

def load_sysp(path=None):
    path = Path(path or SYSP_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): path.write_text(DEFAULT_SYSTEM_PROMPT)
    return path.read_text()
