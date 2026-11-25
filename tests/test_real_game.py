"""Тесты ошибок real_game без запуска браузера."""

from __future__ import annotations

import pytest

pytest.importorskip("playwright")

from dino_master.real_game import is_target_closed


class TargetClosedError(Exception):
    """Имитация playwright TargetClosedError по имени класса."""


class OtherError(Exception):
    """Любая другая ошибка."""


def test_is_target_closed_by_type_name() -> None:
    assert is_target_closed(TargetClosedError("closed"))
    assert not is_target_closed(OtherError("boom"))
