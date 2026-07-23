"The backend table: names, aliases, and the AsyncChat routing kwargs each spec carries."
import pytest
from ipyai.backends import BACKENDS, backend_spec, normalize_backend_name
from ipyai.config import load_config

def test_names_and_aliases():
    assert set(BACKENDS) == {'codex', 'claude', 'claude-api'}
    assert normalize_backend_name() == 'codex'
    assert normalize_backend_name('codex-api') == 'codex'      # legacy names still resolve
    assert normalize_backend_name('claude-cli') == 'claude'
    assert normalize_backend_name('anthropic') == 'claude-api'
    with pytest.raises(ValueError): normalize_backend_name('gemini-cli')

def test_routing_kwargs():
    assert backend_spec('codex').chat_kw == dict(vendor_name='codex')
    assert backend_spec('claude').chat_kw['api_name'] == 'claude_code'
    assert backend_spec('claude-api').chat_kw == dict(vendor_name='anthropic', cache=True)

def test_config_flattens_backend():
    cfg = load_config(backend_name='claude')
    assert cfg['_backend_name'] == 'claude'
    assert cfg['model'] == 'claude-sonnet-4-6'
    assert cfg['completion_model'] == 'claude-haiku-4-5'
