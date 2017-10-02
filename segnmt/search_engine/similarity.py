from typing import Optional

from nltk import edit_distance


def fuzzy_char_level_similarity(x: str, y: str) -> float:
    return 1 - edit_distance(x, y) / max(len(x), len(y))


def fuzzy_word_level_similarity(x: str, y: str, sep: Optional[str] = None):
    xs = x.split(sep)
    ys = y.split(sep)
    return 1 - edit_distance(xs, ys) / max(len(xs), len(ys))
