from pathlib import Path


def flen(file: Path) -> int:
    with open(file) as f:
        _flen = sum(1 for _ in f)
        return _flen
