"Flat config: vendor-prefixed model strings, file over defaults."
import json
from ipyai.config import load_config, DEFAULT_MODEL, DEFAULT_SUGGEST_MODEL, render_sp

def test_defaults(tmp_path):
    p = tmp_path/'config.json'
    cfg = load_config(p)
    assert cfg['model'] == DEFAULT_MODEL and '/' in DEFAULT_MODEL   # vendor-prefixed, not pinned to a model name
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

def test_model_env_overrides_file(tmp_path, monkeypatch):
    "IPYAI_MODEL, when set, chooses the model over both config.json and the default."
    p = tmp_path/'config.json'
    p.write_text(json.dumps(dict(model='anthropic/claude-sonnet-4-6')))
    monkeypatch.setenv('IPYAI_MODEL', 'codex/gpt-x')
    assert load_config(p)['model'] == 'codex/gpt-x'
    monkeypatch.setenv('IPYAI_MODEL', '')
    assert load_config(p)['model'] == 'anthropic/claude-sonnet-4-6'   # empty counts as unset

def test_render_sp():
    s = render_sp('I am {model}; today is {today}.', 'gpt-x')
    assert 'gpt-x' in s and '{' not in s                        # placeholders filled
    s2 = render_sp('no placeholders here', 'gpt-x')
    assert 'You are gpt-x.' in s2 and "Today's date is" in s2   # legacy sp: the lines are appended instead
