"""Tests for observation/action helpers."""

from __future__ import annotations

import pytest

from dino_master import features


def obs(*, obstacle: bool = False, jumping: bool = False) -> dict:
    first = (
        {
            "id": 1,
            "distance": 100.0,
            "type": "CACTUS_SMALL",
            "yPos": 0.0,
            "width": 17.0,
            "speed": 6.0,
        }
        if obstacle
        else {
            "id": None,
            "distance": 1000.0,
            "type": None,
            "yPos": None,
            "width": 0.0,
            "speed": 6.0,
        }
    )
    return {
        "status": "OK",
        "first": first,
        "second": {
            "id": None,
            "distance": 1000.0,
            "type": None,
            "yPos": None,
            "width": 0.0,
            "speed": 6.0,
        },
        "speed": 6.0,
        "dino": {
            "jumping": jumping,
            "ducking": False,
            "yPos": 0.0,
            "speedDrop": False,
        },
    }


def test_valid_actions_mask() -> None:
    assert features.valid_actions(obs()) == [0, 1]
    assert features.valid_actions(obs(obstacle=True)) == [0, 1, 2]
    assert features.valid_actions(obs(obstacle=True, jumping=True)) == [0, 2]


def test_resolve_action_holds_duck_and_jump_cancels() -> None:
    action, remaining = features.resolve_action(2, 0)
    assert action == 2
    assert remaining == features.DUCK_HOLD_STEPS - 1
    action, remaining = features.resolve_action(0, remaining)
    assert action == 2
    assert features.resolve_action(1, 3) == (1, 0)
    assert features.resolve_action(0, 0) == (0, 0)


def test_curriculum_stages() -> None:
    assert features.curriculum_speed(1, 5) == pytest.approx(6.0)
    assert features.curriculum_speed(6, 5) == pytest.approx(8.0)
    assert features.curriculum_speed(11, 5) == pytest.approx(10.0)
    assert features.curriculum_speed(16, 5) == pytest.approx(13.0)
    assert features.curriculum_speed(21, 5) is None


def test_features_length() -> None:
    assert len(features.observation_features(obs())) == features.FEATURE_COUNT == 24
