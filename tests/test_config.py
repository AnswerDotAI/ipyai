"Flat config: vendor-prefixed model strings, file over defaults, -b sugar."
import json
from ipyai.config import load_config, DEFAULT_MODEL, DEFAULT_SUGGEST_MODEL
from ipyai.cli import BACKEND_SUGAR

def test_defaults(tmp_path):
    p = tmp_path/'config.json'
    cfg = load_config(p)
    assert cfg['model'] == DEFAULT_MODEL == 'codex/gpt-5.4'
    assert cfg['suggest_model'] == DEFAULT_SUGGEST_MODEL
    assert json.loads(p.read_text())['model'] == DEFAULT_MODEL   # missing file created with defaults

def test_file_overrides(tmp_path):
    p = tmp_path/'config.json'
    p.write_text(json.dumps(dict(model='anthropic/claude-sonnet-4-6', think='H', backend='claude')))
    cfg = load_config(p)
    assert cfg['model'] == 'anthropic/claude-sonnet-4-6'
    assert cfg['think'] == 'h'                     # normalized
    assert cfg['suggest_model'] == DEFAULT_SUGGEST_MODEL
    assert 'backend' not in cfg                    # pre-flat nested keys are ignored

def test_backend_sugar():
    assert set(BACKEND_SUGAR) == {'codex', 'claude', 'claude-api'}
    for model, suggest in BACKEND_SUGAR.values():
        assert '/' in model and '/' in suggest     # every pair is vendor-prefixed
