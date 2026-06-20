import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict import validate_inputs

def test_valid_inputs():
    values = [90, 42, 43, 25.0, 80.0, 6.5, 200.0]
    assert validate_inputs(values) == []

def test_invalid_nitrogen():
    values = [141, 42, 43, 25.0, 80.0, 6.5, 200.0]
    errors = validate_inputs(values)
    assert len(errors) > 0
    assert "Nitrogen" in errors[0]

def test_invalid_ph():
    values = [90, 42, 43, 25.0, 80.0, 15.0, 200.0]
    errors = validate_inputs(values)
    assert len(errors) > 0
    assert "pH" in errors[0]

def test_lower_boundaries():
    values = [0, 5, 5, 8.0, 14.0, 3.5, 20.0]
    assert validate_inputs(values) == []

def test_upper_boundaries():
    values = [140, 145, 205, 44.0, 100.0, 9.95, 299.0]
    assert validate_inputs(values) == []