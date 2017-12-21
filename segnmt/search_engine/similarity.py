from typing import Callable
from typing import Dict
from typing import List

from nltk import edit_distance
from nltk.translate import bleu_score


def char_level_edit_distance(x: str, y: str) -> float:
    return 1 - edit_distance(x, y) / max(len(x), len(y))


def word_level_edit_distance(x: str, y: str) -> float:
    xs = x.split()
    ys = y.split()
    return 1 - edit_distance(xs, ys) / max(len(xs), len(ys))


def bleu(x: str, y: str) -> float:
    list_of_references: List[List[List[str]]] = [[x.split()]]
    hypotheses: List[List[str]] = [y.split()]
    return bleu_score.corpus_bleu(
        list_of_references,
        hypotheses
    )


functions: Dict[str, Callable[[str, str], float]] = {
    'edit-word': word_level_edit_distance,
    'edit-char': char_level_edit_distance,
    'bleu': bleu
}
