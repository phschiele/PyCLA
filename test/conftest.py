from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture(scope="session", name="resource_dir")
def resource_fixture():
    """resource fixture"""
    return Path(__file__).parent / "data_test"


@pytest.fixture()
def update_tests() -> Generator[bool, None, None]:
    update = False
    yield update
    assert not update, "Set to false after updating tests."
