from typing import Callable
from typing import Dict
from typing import List

from nltk import edit_distance
from nltk.translate import bleu_score
from pyknp import Juman


juman = Juman()
N = 100


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


def _hinshi(s: str) -> str:
    result = juman.analysis("".join(s.strip().split()))
    return " ".join([m.hinsi for m in result.mrph_list()])


def jiritsugo_edit_distance(x: str, y: str) -> float:
    x_j = _jiritsugo(x)
    y_j = _jiritsugo(y)
    return word_level_edit_distance(x_j, y_j)


def hinshi_edit_distance(x: str, y: str) -> float:
    x_h = _hinshi(x)
    y_h = _hinshi(y)
    return word_level_edit_distance(x_h, y_h)


def ngram_coverage(x: str, y: str) -> float:
    x_1 = x.strip().split()
    x_2 = list(zip(x_1, x_1[1:]))
    x_3 = list(zip(x_1, x_1[1:], x_1[2:]))
    x_4 = list(zip(x_1, x_1[1:], x_1[2:], x_1[3:]))

    if len(x_1) == 0:
        return 0.0

    y_1 = y.strip().split()
    y_2 = list(zip(y_1, y_1[1:]))
    y_3 = list(zip(y_1, y_1[1:], y_1[2:]))
    y_4 = list(zip(y_1, y_1[1:], y_1[2:], y_1[3:]))

    if len(y_1) == 0:
        return 0.0

    p_1 = sum(min(x_1.count(gram), y_1.count(gram)) for gram in set(x_1))
    p_2 = sum(min(x_2.count(gram), y_2.count(gram)) for gram in set(x_2)) + 1
    p_3 = sum(min(x_3.count(gram), y_3.count(gram)) for gram in set(x_3)) + 1
    p_4 = sum(min(x_4.count(gram), y_4.count(gram)) for gram in set(x_4)) + 1

    a_1 = p_1 * 1.0 / len(x_1)
    a_2 = p_2 * 1.0 / (len(x_2) + 1.0)
    a_3 = p_3 * 1.0 / (len(x_3) + 1.0)
    a_4 = p_4 * 1.0 / (len(x_4) + 1.0)

    return a_1 ** 0.25 * a_2 ** 0.25 * a_3 ** 0.25 * a_4 ** 0.25


def match_length(x: str, y: str) -> float:
    xs = x.strip().split()
    substrs_list: List[List[str]] = [
        [
            ' '.join(xs[start:start+length])
            for start in range(len(xs) - length + 1)
        ]
        for length in range(1, len(xs) + 1)
    ]
    score = sum([
        sum([y.count(substr) * (N ** i) for substr in substrs])
        for i, substrs in enumerate(substrs_list)
    ])
    return float(score)


functions: Dict[str, Callable[[str, str], float]] = {
    'edit-word': word_level_edit_distance,
    'edit-char': char_level_edit_distance,
    'bleu': bleu,
    'no-change': no_change,
    'edit-jiritsugo': jiritsugo_edit_distance,
    'edit-hinshi': hinshi_edit_distance,
    'ngram-coverage': ngram_coverage,
    'match-length': match_length
}
