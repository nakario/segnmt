from abc import ABCMeta
from abc import abstractmethod
from itertools import chain
from typing import Callable
from typing import Iterable
from typing import List
from typing import Tuple


class BaseEngine(metaclass=ABCMeta):
    @abstractmethod
    def search(self, sentence: str) -> List[Tuple[str, str, str]]:
        raise NotImplementedError()


class Retriever:
    def __init__(self,
                 engine: BaseEngine,
                 similarity: Callable[[str, str], float],
                 limit: int = -1,
                 training: bool = False):
        self.engine = engine
        self.similarity = similarity
        self.limit = limit
        self.training = training

    def retrieve(self, src: str, index: int) -> List[Tuple[str, str, str]]:
        pairs = self.engine.search(src)
        if self.training:
            pairs = filter(lambda x: x[0] != str(index), pairs)

        reranked_pairs = self.rerank(pairs, src)

        # Return the top-K similar pairs
        if self.limit >= 0:
            return list(reranked_pairs)[:self.limit]

        # Return greedy selected pairs based on the coverage of symbols
        retrieved = []
        coverage = 0.0
        src_symbols = src.strip().split(" ")
        for pair in reranked_pairs:
            sentences = [pair_[1] for pair_ in retrieved] + [pair[1]]
            symbols = flatten([s.split(" ") for s in sentences])
            c_tmp = sum([s in symbols for s in src_symbols]) / len(src_symbols)
            if c_tmp > coverage:
                coverage = c_tmp
                retrieved.append(pair)
        return retrieved

    def rerank(self, pairs: Iterable[Tuple[str, str, str]], src: str) \
            -> Iterable[Tuple[str, str, str]]:
        return sorted(pairs,
                      reverse=True,
                      key=lambda pair: self.similarity(src, pair[1]))


def flatten(x: Iterable[Iterable[str]]) -> List[str]:
    return list(chain.from_iterable(x))
