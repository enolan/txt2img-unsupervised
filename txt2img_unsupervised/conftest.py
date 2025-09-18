import sys

import pytest

from .gpu_check import gpu_is_ampere_or_newer


@pytest.fixture
def starts_with_progressbar(request):
    """Avoid tqdm progress bars erasing test names when running verbose pytest in a TTY."""
    config = request.config
    if config.getoption("verbose") <= 0:
        return

    terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")
    if not terminal_reporter:
        return

    writer = getattr(terminal_reporter, "_tw", None)
    wrote_newline = False
    if writer and getattr(writer, "isatty", False):
        terminal_reporter.ensure_newline()
        terminal_reporter.write_line("")
        wrote_newline = True

    # tqdm defaults to stderr; ensure we also start on a fresh line there.
    if sys.stderr is not None and sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_ampere_or_newer: mark test as needing Ampere or newer GPU microarchitecture",
    )


def pytest_runtest_setup(item):
    if "requires_ampere_or_newer" in item.keywords:
        if not gpu_is_ampere_or_newer():
            pytest.skip("Test requires Ampere or newer GPU microarchitecture")
