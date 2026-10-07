import asyncio
from collections import deque
from aidialog.msg_parts import Text

async def until(pred, timeout=25):
    async with asyncio.timeout(timeout):
        while not pred(): await asyncio.sleep(0.02)

async def send(app, text):
    app.comp.on_bytes(b'\x1b[200~' + text.encode() + b'\x1b[201~\r')
    await until(lambda: not app.busy and app._pending is None and not app.buf.text)

class ScriptedModel:
    "Script model streams/final replies, leaving the app, kernel, dialog and terminal real."
    def __init__(self): self.turns, self.requests = deque(), []
    def queue(self, reply='', events=None, gate=None): self.turns.append((reply, [Text(reply)] if events is None else events, gate))
    def __call__(self, **settings): return ScriptedChat(self, settings)

class ScriptedChat:
    def __init__(self, model, settings): self.model, self.settings = model, settings
    async def __call__(self, msg=None, **kwargs):
        self.model.requests.append(dict(self.settings, msg=msg))
        self.reply, events, gate = self.model.turns.popleft()
        async def stream():
            for event in events:
                await asyncio.sleep(0)
                if isinstance(event, Exception): raise event
                yield event
            if gate is not None: await gate.wait()
        return stream()
    def full(self, **kwargs): return self.reply
