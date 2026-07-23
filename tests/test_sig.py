from ipyai.sig import call_context, parse_sig_text, active_param

def test_call_context():
    assert call_context('print(', 6) == (5, 0)
    assert call_context('print(1, ', 9) == (5, 1)
    assert call_context('f(g(x, y), ', 11) == (1, 1)   # inner call closed: back in f, one comma
    assert call_context('f(g(x, ', 7) == (3, 1)        # innermost open call is g
    assert call_context('x = 1 + 2', 9) is None
    assert call_context('d["a,b"](', 9) == (8, 0)      # commas inside strings ignored; subscript callable
    assert call_context('(a, b)', 3) is None           # grouping parens are not a call
    assert call_context('np.arange(1, 2', 14) == (9, 1)

def test_parse_sig_text():
    blob = ('Signature: arange(start, stop=None, step=1, *, dtype=None)\n'
            'Docstring: Return evenly spaced values.\nType: builtin')
    name, params, doc = parse_sig_text(blob)
    assert name == 'arange'
    assert params == ['start', 'stop=None', 'step=1', '*', 'dtype=None']
    assert doc == 'Return evenly spaced values.'
    assert parse_sig_text('Type: module') is None

def test_active_param():
    ps = ['a', 'b=1', '*args', 'c=2']
    assert active_param(ps, 0) == 0
    assert active_param(ps, 2) == 2
    assert active_param(ps, 5) == 2   # *args soaks the overflow
    assert active_param(['a'], 3) is None
