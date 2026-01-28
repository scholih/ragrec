"""Tests for ETL module."""

import pytest

from ragrec.etl.hm_loader import create_age_bracket


def test_create_age_bracket_under_20() -> None:
    """Test age bracket for users under 20."""
    assert create_age_bracket(15) == "under_20"
    assert create_age_bracket(19) == "under_20"


def test_create_age_bracket_20_29() -> None:
    """Test age bracket for users 20-29."""
    assert create_age_bracket(20) == "20-29"
    assert create_age_bracket(25) == "20-29"
    assert create_age_bracket(29) == "20-29"


def test_create_age_bracket_30_39() -> None:
    """Test age bracket for users 30-39."""
    assert create_age_bracket(30) == "30-39"
    assert create_age_bracket(35) == "30-39"
    assert create_age_bracket(39) == "30-39"


def test_create_age_bracket_40_49() -> None:
    """Test age bracket for users 40-49."""
    assert create_age_bracket(40) == "40-49"
    assert create_age_bracket(45) == "40-49"
    assert create_age_bracket(49) == "40-49"


def test_create_age_bracket_50_59() -> None:
    """Test age bracket for users 50-59."""
    assert create_age_bracket(50) == "50-59"
    assert create_age_bracket(55) == "50-59"
    assert create_age_bracket(59) == "50-59"


def test_create_age_bracket_60_plus() -> None:
    """Test age bracket for users 60+."""
    assert create_age_bracket(60) == "60+"
    assert create_age_bracket(75) == "60+"
    assert create_age_bracket(100) == "60+"


def test_create_age_bracket_none() -> None:
    """Test age bracket for None value."""
    assert create_age_bracket(None) is None
