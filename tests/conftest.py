import asyncio, os, sqlite3, tempfile
from types import SimpleNamespace

import pytest

import ipyai.core as core
from ipyai.kernel_bridge import KernelBridge


_IPYTHONDIR_SESSION = None


def pytest_configure(config):
    "Redirect IPYTHONDIR for the whole test session so no test run pollutes the user's real ~/.ipython."
    global _IPYTHONDIR_SESSION
    _IPYTHONDIR_SESSION = tempfile.mkdtemp(prefix="ipyai-test-ipy-")
    os.environ["IPYTHONDIR"] = _IPYTHONDIR_SESSION


def pytest_unconfigure(config):
    import shutil
    if _IPYTHONDIR_SESSION: shutil.rmtree(_IPYTHONDIR_SESSION, ignore_errors=True)


def _make_test_db():
    "Create an isolated sqlite DB shaped like IPython's history.sqlite plus our claude_prompts table, for DummyShell-based tests."
    db = sqlite3.connect(":memory:")
    with db:
        db.execute("CREATE TABLE sessions (session INTEGER PRIMARY KEY AUTOINCREMENT, "
            "start TIMESTAMP DEFAULT CURRENT_TIMESTAMP, end TIMESTAMP, num_cmds INTEGER DEFAULT 0, remark TEXT)")
        db.execute("CREATE TABLE history (session INTEGER, line INTEGER, source TEXT, source_raw TEXT, PRIMARY KEY (session, line))")
        db.execute(core._PROMPTS_SQL)
    return db


class DummyDisplayPublisher:
    def __init__(self): self._is_publishing = False


class DummyInputTransformerManager:
    def __init__(self): self.cleanup_transforms = []


class DummyHistory:
    "Minimal client-side history stand-in used by DummyShell tests."
    def __init__(self, session_number=1):
        self.session_number = session_number
        self.entries = {}
        self.input_hist_parsed = [""]
        self.input_hist_raw = [""]

    def add(self, line, source, output=None): self.entries[line] = (source, output)

    def get_range(self, session=0, start=1, stop=None, raw=True, output=False):
        if stop is None: stop = max(self.entries, default=0) + 1
        for i in range(start, stop):
            if i not in self.entries: continue
            src,out = self.entries[i]
            yield (0, i, (src, out) if output else src)


class DummyShell:
    "In-process stand-in for IPyAIShell used by unit tests that don't need a real kernel."
    def __init__(self):
        self.input_transformer_manager = DummyInputTransformerManager()
        self.user_ns = {}
        self.magics = []
        self.history_manager = DummyHistory()
        self.display_pub = DummyDisplayPublisher()
        self.execution_count = 1
        self.ran_cells = []
        self.output_buffer = {}
        self.prompts = SimpleNamespace(in_prompt_tokens=lambda: [])

    def register_magics(self, magics): self.magics.append(magics)
    def set_custom_exc(self, *args): pass

    def run_cell(self, source, store_history=False):
        self.ran_cells.append((source, store_history))
        try: exec(compile(source, f"<cell-{self.execution_count}>", "exec"), self.user_ns)
        except Exception as e: return SimpleNamespace(success=False, error_in_exec=e, result=None)
        if store_history:
            self.history_manager.add(self.execution_count, source)
            self.execution_count += 1
        return SimpleNamespace(success=True, result=None, error_in_exec=None, error_before_exec=None)

    async def run_cell_async(self, source, store_history=False, transformed_cell=None):
        return self.run_cell(transformed_cell or source, store_history=store_history)


@pytest.fixture(autouse=True)
def temp_core_paths(tmp_path, monkeypatch):
    cfg = tmp_path/"config"
    cfg.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(core, "CONFIG_DIR", cfg)
    monkeypatch.setattr(core, "CONFIG_PATH", cfg/"config.json")
    monkeypatch.setattr(core, "SYSP_PATH", cfg/"sysp.txt")
    monkeypatch.setattr(core, "LOG_PATH", cfg/"exact-log.jsonl")
    yield


@pytest.fixture
def shell(): return DummyShell()


@pytest.fixture
def test_db():
    "Isolated in-memory sqlite shaped like the shared history DB."
    db = _make_test_db()
    # Pre-allocate session 1 so IPyAIController has something to reference.
    with db: db.execute("INSERT INTO sessions (session) VALUES (1)")
    yield db
    db.close()


_KERNEL_BOOTSTRAP = ("from IPython import get_ipython\n"
    "_ip = get_ipython()\n"
    "try: _ip.extension_manager.load_extension('safepyrun')\n"
    "except Exception: pass\n"
    "try: _ip.extension_manager.load_extension('ipythonng')\n"
    "except Exception: pass\n"
    "_ip.history_manager.db_log_output = True\n")


async def _prepare_kernel_bridge(client):
    bridge = KernelBridge(client)
    await bridge._exec(_KERNEL_BOOTSTRAP)
    await bridge.inject_tools()
    await bridge.available_names(force=True)
    return bridge


async def _snapshot_globals(bridge):
    exprs,_ = await bridge._exec("", expressions={"_r": "[k for k in globals() if not k.startswith('_')]"})
    return set(exprs.get("_r") or [])


async def _clear_extras(bridge, baseline):
    exprs,_ = await bridge._exec("", expressions={
        "_r": "[k for k in globals() if not k.startswith('_') and k not in %r]" % list(baseline)})
    extras = exprs.get("_r") or []
    if extras: await bridge._exec("\n".join(f"globals().pop({n!r}, None)" for n in extras))


@pytest.fixture(scope="session")
def session_kernel():
    from jupyter_client.manager import KernelManager
    from jupyter_client.asynchronous.client import AsyncKernelClient
    km = KernelManager()
    km.start_kernel(extra_arguments=["--HistoryManager.enabled=True"])
    loop = asyncio.new_event_loop()

    async def _setup():
        client = AsyncKernelClient()
        client.load_connection_file(km.connection_file)
        client.start_channels()
        await client.wait_for_ready(timeout=30)
        bridge = await _prepare_kernel_bridge(client)
        baseline = await _snapshot_globals(bridge)
        return client, bridge, baseline

    client,bridge,baseline = loop.run_until_complete(_setup())
    try: yield dict(manager=km, client=client, bridge=bridge, baseline=baseline, loop=loop)
    finally:
        try: loop.run_until_complete(client.stop_channels())
        except Exception:
            try: client.stop_channels()
            except Exception: pass
        try: km.shutdown_kernel(now=False)
        except Exception: pass
        try: loop.close()
        except Exception: pass


@pytest.fixture
def kernel_bridge(session_kernel, request):
    "Session kernel bridge with per-test teardown that clears any user_ns names the test added."
    bridge = session_kernel["bridge"]
    baseline = session_kernel["baseline"]
    loop = session_kernel["loop"]

    def _finalize():
        try: loop.run_until_complete(_clear_extras(bridge, baseline))
        except Exception: pass

    request.addfinalizer(_finalize)
    return bridge


@pytest.fixture
def kernel_loop(session_kernel): return session_kernel["loop"]
