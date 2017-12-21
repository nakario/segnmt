from typing import Callable
from typing import Dict
from typing import List

from nltk import edit_distance
from nltk.translate import bleu_score
from pyknp import Juman


juman = Juman()


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


def no_change(_x: str, _y: str) -> float:
    return 1.0


def _jiritsugo(s: str) -> str:
    result = juman.analysis("".join(s.strip().split()))
    return " ".join([
        m.genkei for m in result.mrph_list()
        if m.hinsi in ["名詞", "動詞", "形容詞", "未定義語"]
    ])


def jiritsugo_edit_distance(x: str, y: str) -> float:
    x_j = _jiritsugo(x)
    y_j = _jiritsugo(y)
    return word_level_edit_distance(x_j, y_j)


functions: Dict[str, Callable[[str, str], float]] = {
    'edit-word': word_level_edit_distance,
    'edit-char': char_level_edit_distance,
    'bleu': bleu,
    'no-change': no_change,
    'edit-jiritsugo': jiritsugo_edit_distance
}
