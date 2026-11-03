"""Shared access to the quick-start script's example inputs."""

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def quickstart():
    path = Path(__file__).resolve().parents[1] / "scripts/patch_to_ecsv.py"
    spec = importlib.util.spec_from_file_location("patch_to_ecsv", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
