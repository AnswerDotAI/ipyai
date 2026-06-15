"End-to-end test that actually runs IPyAIApp.initialize() against a real spawned kernel. Stops short of starting the interactive prompt loop but verifies every wiring step (kernel spawn, async client, shell subclass, bootstrap inject, controller attach)."


def _patch_core(tmp_path, monkeypatch):
    import ipyai.core as core
    monkeypatch.setattr(core, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(core, "CONFIG_PATH", tmp_path/"config.json")
    monkeypatch.setattr(core, "SYSP_PATH", tmp_path/"sysp.txt")
    monkeypatch.setattr(core, "LOG_PATH", tmp_path/"exact-log.jsonl")


def test_init_banner_skipped_when_attaching_to_existing_kernel():
    "The jupyter_console banner (Python/IPython version + tip) is noise when attaching — user already knows the kernel. Suppress it for --existing."
    from types import SimpleNamespace
    from ipyai.app import IPyAIApp

    called = []
    app = IPyAIApp.__new__(IPyAIApp)
    app.shell = SimpleNamespace(show_banner=lambda: called.append("shown"))

    app.existing = "/tmp/kernel-abc.json"
    app.init_banner()
    assert called == [], "banner must be suppressed when attaching to an existing kernel"

    app.existing = ""
    app.init_banner()
    assert called == ["shown"], "banner must still print for a kernel we spawned"


def test_app_initialize_wires_full_stack(tmp_path, monkeypatch):
    _patch_core(tmp_path, monkeypatch)

    from ipyai.app import IPyAIApp
    from ipyai.shell import IPyAIShell

    IPyAIShell.clear_instance()
    app = IPyAIApp()
    try:
        app.initialize(["--simple-prompt"])

        assert isinstance(app.shell, IPyAIShell), f"shell is {type(app.shell).__name__}, expected IPyAIShell"
        assert app._bridge is not None
        assert app.shell._ipyai_bridge is app._bridge

        cmd = app.kernel_manager.format_kernel_cmd()
        assert "-Xfrozen_modules=off" in cmd, f"kernel cmd must disable frozen modules for debugger: {cmd}"
        assert cmd.index("-Xfrozen_modules=off") < cmd.index("-m"), f"flag must precede -m: {cmd}"

        ctrl = app.shell._ipyai_controller
        assert ctrl is not None, "ipyai controller should have loaded"
        assert ctrl.loaded

        import asyncio
        async def _check():
            names = set(await app._bridge.available_names(force=True))
            return names
        names = asyncio.get_event_loop().run_until_complete(_check())
        assert "python" in names, f"python missing from kernel ns: {names}"
    finally:
        try: app._finalize_kernel()
        except Exception: pass
        try:
            if app.kernel_manager and app.kernel_manager.is_alive(): app.kernel_manager.shutdown_kernel(now=True)
        except Exception: pass
        IPyAIShell.clear_instance()
