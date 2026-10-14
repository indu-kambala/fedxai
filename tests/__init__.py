from pathlib import Path


def test_placeholder() -> None:
    assert Path(__file__).is_file()
