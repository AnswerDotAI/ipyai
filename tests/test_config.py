"Flat config: vendor-prefixed model strings, file over defaults."
import json
from ipyai.config import load_config, DEFAULT_MODEL, DEFAULT_SUGGEST_MODEL

def test_defaults(tmp_path):
    p = tmp_path/'config.json'
    cfg = load_config(p)
    assert cfg['model'] == DEFAULT_MODEL and '/' in DEFAULT_MODEL   # vendor-prefixed, not pinned to a model name
    assert cfg['suggest_model'] == DEFAULT_SUGGEST_MODEL
    assert cfg['prompt_style'] == '' and cfg['pad_transcript'] is False   # transcript styling opt-in
    assert json.loads(p.read_text())['model'] == DEFAULT_MODEL   # missing file created with defaults

def test_file_overrides(tmp_path):
    p = tmp_path/'config.json'
    p.write_text(json.dumps(dict(model='anthropic/claude-sonnet-4-6', think='H', backend='claude', prompt_style='bold magenta', pad_transcript=1)))
    cfg = load_config(p)
    assert cfg['model'] == 'anthropic/claude-sonnet-4-6'
    assert cfg['think'] == 'h'                     # normalized
    assert cfg['suggest_model'] == DEFAULT_SUGGEST_MODEL
    assert 'backend' not in cfg                    # pre-flat nested keys are ignored
    assert cfg['prompt_style'] == 'bold magenta'
    assert cfg['pad_transcript'] is True
