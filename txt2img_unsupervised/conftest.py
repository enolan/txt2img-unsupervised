import pytest

from .gpu_check import gpu_is_ampere_or_newer

def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_ampere_or_newer: mark test as needing Ampere or newer GPU microarchitecture"
    )


def pytest_runtest_setup(item):
    if "requires_ampere_or_newer" in item.keywords:
        if not gpu_is_ampere_or_newer():
            pytest.skip("Test requires Ampere or newer GPU microarchitecture")