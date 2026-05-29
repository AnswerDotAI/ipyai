"IPyAIApp — ZMQTerminalIPythonApp subclass wiring ipyai flags, our IPyAIShell, and kernel-side setup (extensions config + tool injection)."
import asyncio, os, signal, sys

from jupyter_client.asynchronous.client import AsyncKernelClient
from jupyter_client.manager import KernelManager
from jupyter_console.app import ZMQTerminalIPythonApp, aliases as _JC_ALIASES, flags as _JC_FLAGS
from traitlets import Bool, Dict, Int, Type, Unicode

from .backends import DEFAULT_BACKEND
from .kernel_bridge import CUSTOM_TOOL_NAMES, KernelBridge
from .shell import IPyAIShell


_HELP_EPILOG = """
ipyai flags (in addition to standard jupyter_console flags):
  -b BACKEND        AI backend: codex-api (default), codex, claude-cli, claude-api
  -r N              resume ipyai session N
  --resume-pick     pick an ipyai session to resume interactively
  -l FILE           load a notebook (.ipynb) at startup
  -p                start in prompt mode
  --keep-alive      do not shut down the kernel on exit (prints connection file)
"""


def _preprocess_argv(argv):
    "Turn bare `-r` (no number) into `--resume-pick`; leave `-r N` alone."
    out, i = [], 0
    while i < len(argv):
        if argv[i] == "-r":
            if i+1 < len(argv) and argv[i+1].lstrip("-").isdigit():
                out.extend([argv[i], argv[i+1]])
                i += 2
            else:
                out.append("--resume-pick")
                i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


def _open_async_client(connection_file):
    "Open an AsyncKernelClient against an existing connection file. Starts channels."
    client = AsyncKernelClient()
    client.load_connection_file(connection_file)
    client.start_channels()
    return client


class IPyAIKernelManager(KernelManager):
    "KernelManager that disables frozen modules so debugpy doesn't warn and breakpoints work."
    def format_kernel_cmd(self, extra_arguments=None):
        cmd = super().format_kernel_cmd(extra_arguments)
        if cmd and "-Xfrozen_modules=off" not in cmd: cmd = [cmd[0], "-Xfrozen_modules=off", *cmd[1:]]
        return cmd


class IPyAIApp(ZMQTerminalIPythonApp):
    name = "ipyai"
    classes = ZMQTerminalIPythonApp.classes + [IPyAIShell]
    description = "Terminal IPython with AI backends, driving a Jupyter kernel via ZMQ."
    examples = _HELP_EPILOG

    kernel_manager_class = Type(IPyAIKernelManager, config=True, help="Kernel manager that injects -Xfrozen_modules=off")

    backend = Unicode(DEFAULT_BACKEND, config=True, help="AI backend name")
    resume_session = Int(0, config=True, help="Resume the given ipyai session id (0 = none)")
    resume_pick = Bool(False, config=True, help="Pick an ipyai session to resume interactively")
    load_notebook_path = Unicode("", config=True, help="Notebook path to load at startup")
    prompt_mode = Bool(False, config=True, help="Start in prompt mode")
    keep_alive = Bool(False, config=True, help="Leave the kernel running when ipyai exits")

    aliases = Dict(dict(_JC_ALIASES, b="IPyAIApp.backend", r="IPyAIApp.resume_session", l="IPyAIApp.load_notebook_path"))
    flags = Dict(dict(_JC_FLAGS, p=({"IPyAIApp": {"prompt_mode": True}}, "Start in prompt mode"),
        **{"resume-pick": ({"IPyAIApp": {"resume_pick": True}}, "Pick an ipyai session to resume"),
            "keep-alive": ({"IPyAIApp": {"keep_alive": True}}, "Leave the kernel running on exit")}))

    def initialize(self, argv=None):
        if argv is None: argv = sys.argv[1:]
        argv = _preprocess_argv(argv)
        super().initialize(argv=argv)

    def init_banner(self):
        "Suppress the Python/IPython banner when attaching — user already knows what they connected to."
        if self.existing: return
        super().init_banner()

    def build_kernel_argv(self, argv=None):
        "Force-enable the kernel's HistoryManager so we have a real sqlite DB to share; users with HistoryManager.enabled=False in their IPython config still get ipyai's history features."
        super().build_kernel_argv(argv)
        self.kernel_argv = list(self.kernel_argv) + ["--HistoryManager.enabled=True"]

    def init_kernel_client(self):
        super().init_kernel_client()
        cf = getattr(self.kernel_client, "connection_file", "") or getattr(self.kernel_manager, "connection_file", "")
        self._async_client = _open_async_client(cf)
        self._bridge = KernelBridge(self._async_client)

    def init_shell(self):
        from jupyter_client.consoleapp import JupyterConsoleApp
        JupyterConsoleApp.initialize(self)
        signal.signal(signal.SIGINT, self.handle_sigint)
        self.shell = IPyAIShell.instance(parent=self, manager=self.kernel_manager,
            client=self.kernel_client, confirm_exit=self.confirm_exit)
        # jupyter_console.mainloop sets `keepkernel = not own_kernel` unconditionally then uses that
        # to decide whether to call client.shutdown(). With --keep-alive we lie that we don't own
        # the kernel, so mainloop skips the shutdown and leaves it running.
        self.shell.own_kernel = (not self.existing) and (not self.keep_alive)
        self.shell._ipyai_bridge = self._bridge
        self.shell.install_iopub_tee()
        hist_path, session_number = self._bootstrap_kernel()
        self._db_path, self._session_number = hist_path, session_number
        self._load_ipyai_controller()
        self._install_history_adapter()

    def _kernel_startup_code(self):
        exts = ("safepyrun", "ipythonng")
        lines = ["from IPython import get_ipython", "_ip = get_ipython()"]
        for ext in exts: lines.append(f"try: _ip.extension_manager.load_extension({ext!r})\nexcept Exception: pass")
        lines.append("_ip.history_manager.db_log_output = True")
        return "\n".join(lines)

    def _bootstrap_kernel(self):
        async def _go():
            await self._bridge._exec(self._kernel_startup_code())
            await self._bridge.inject_tools()
            await self._bridge.available_names(force=True)
            return await self._bridge.history_db_info()
        return asyncio.get_event_loop().run_until_complete(_go())

    def _load_ipyai_controller(self):
        from .core import create_controller, _open_db
        resume = -1 if self.resume_pick else (self.resume_session or None)
        db = _open_db(self._db_path)
        existing = self.connection_file if self.existing else None
        ctrl = create_controller(self.shell, resume=resume, file=self.load_notebook_path or None, prompt_mode=self.prompt_mode,
            backend=self.backend, bridge=self._bridge, db=db, session_number=self._session_number, existing=existing)
        self.shell._ipyai_controller = ctrl

    def _install_history_adapter(self):
        from .shell import IPyAIHistory
        pt = getattr(self.shell, "pt_cli", None)
        if pt is None: return
        ctrl = self.shell._ipyai_controller
        hist = IPyAIHistory(ctrl.db, int(self._session_number or 0), mode_fn=lambda: "prompt" if ctrl.prompt_mode else "code")
        pt.default_buffer.history = hist

    def start(self):
        self._install_signal_handlers()
        try: super().start()
        finally: self._finalize_kernel()

    def _install_signal_handlers(self):
        for sig in (signal.SIGHUP, signal.SIGTERM):
            try: signal.signal(sig, self._signal_exit)
            except (ValueError, OSError): pass

    def _signal_exit(self, signum, frame):
        self._finalize_kernel()
        os._exit(0)

    def _finalize_kernel(self):
        "Client-side teardown only. jupyter_console.mainloop handles the actual kernel shutdown (via client.shutdown) or keeps it alive based on own_kernel / keepkernel."
        try: self._async_client.stop_channels()
        except Exception: pass
        if self.keep_alive and not self.existing:
            path = getattr(self.kernel_manager, "connection_file", "")
            if path: print(f"Kernel still running. Connection file: {path}")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_KEY"): os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_KEY"]
    IPyAIApp.launch_instance()


if __name__ == "__main__": main()
