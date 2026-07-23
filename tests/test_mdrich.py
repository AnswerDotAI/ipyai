from rich.syntax import Syntax
from rich.text import Text
from ipyai.mdrich import md_blocks

MD = '''# Title

Some **bold**, `code`, and [a link](http://x.com).

```python
print(1)
```

- one
- two
  - nested

> quoted line

| a | b |
|---|---|
| 1 | 2 |
'''

def test_md_blocks():
    bs = md_blocks(MD)
    assert bs[0].plain == '# Title'
    assert any(s.style and 'bold' in str(s.style) for s in bs[1].spans)
    assert 'a link (http://x.com)' in bs[1].plain
    assert isinstance(bs[2], Syntax) and bs[2].code == 'print(1)'
    assert '• one' in bs[3].plain and '  • nested' in bs[3].plain
    assert bs[4].plain.startswith('│ quoted')
    assert 'a │ b' in bs[5].plain and '1 │ 2' in bs[5].plain
    assert all(hasattr(b, '__rich_console__') or isinstance(b, Text) for b in bs)
