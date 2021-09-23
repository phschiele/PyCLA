from typing import Generator

import pytest


@pytest.fixture()
def path_test_data() -> str:
    return "test/data_test/"


@pytest.fixture()
def update_tests() -> Generator[bool, None, None]:
    update = False
    yield update
    assert not update, "Set to false after updating tests."
