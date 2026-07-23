"""Backend table: each backend is a set of fastllm `AsyncChat` routing kwargs plus default models.

All backends ride one fastllm code path (assistant.py). `codex` uses fastllm's native codex vendor
(chatgpt backend-api completions, token from ~/.codex/auth.json), which subsumes both the old
app-server CLI client and the litellm chatgpt/ aliasing. `claude` uses fastllm-claude-code's
registered `claude_code` api (Claude Code subscription; imported lazily at chat construction).
`claude-api` is the plain Anthropic API with prompt caching."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class BackendSpec:
    name: str
    label: str
    chat_kw: dict = field(default_factory=dict)  # AsyncChat routing kwargs (vendor_name/api_name/cache)
    default_model: str = ''
    default_completion_model: str = ''

BACKENDS = {
    'codex': BackendSpec('codex', 'Codex (ChatGPT subscription)', dict(vendor_name='codex'),
                         'gpt-5.4', 'gpt-5.4-mini'),
    'claude': BackendSpec('claude', 'Claude (Claude Code subscription)',
                          dict(api_name='claude_code', vendor_name='claude_code'),
                          'claude-sonnet-4-6', 'claude-haiku-4-5'),
    'claude-api': BackendSpec('claude-api', 'Claude API', dict(vendor_name='anthropic', cache=True),
                              'claude-sonnet-4-6', 'claude-haiku-4-5'),
}
DEFAULT_BACKEND = 'codex'
BACKEND_ALIASES = {'codex-api': 'codex', 'claude-cli': 'claude', 'cli': 'claude', 'api': 'claude-api',
                   'anthropic': 'claude-api'}

def normalize_backend_name(name=None):
    key = (name or DEFAULT_BACKEND).strip().lower()
    if key in BACKENDS: return key
    if key in BACKEND_ALIASES: return BACKEND_ALIASES[key]
    raise ValueError(f"Unknown backend {name!r}. Expected one of: {', '.join(BACKENDS)}")

def backend_spec(name=None): return BACKENDS[normalize_backend_name(name)]
