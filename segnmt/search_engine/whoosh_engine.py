from logging import getLogger
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

from progressbar import ProgressBar
from whoosh.fields import NGRAM
from whoosh.fields import Schema
from whoosh.fields import STORED
from whoosh.index import Index
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.qparser import OrGroup
from whoosh.qparser import QueryParser

from segnmt.misc.functions import flen
from segnmt.search_engine.retriever import BaseEngine


logger = getLogger(__name__)


class WhooshEngine(BaseEngine):
    def __init__(self, limit, index):
        self.limit = limit
        self.index = index
        self.parser = QueryParser("SRC", index.schema, group=OrGroup)

    def search(self, sentence: str) -> List[Tuple[str, str, str]]:
        with self.index.searcher() as searcher:
            query = self.parser.parse(sentence)
            results = searcher.search(query, limit=self.limit)
            return [(r["ID"], r["SRC"], r["TGT"]) for r in results]


def create_index(index_path: Path,
                 source: Path,
                 target: Path) -> Index:
    assert source.exists()
    assert target.exists()
    sentence_count = flen(source)
    assert flen(target) == sentence_count
    index_path.mkdir(parents=True)
    schema = Schema(ID=STORED, SRC=NGRAM(stored=True), TGT=STORED)
    ix = create_in(index_path.absolute(), schema)
    writer = ix.writer()
    bar = ProgressBar(max_value=sentence_count)
    logger.info(f'Creating index at {index_path.absolute()}')
    with open(source) as src, open(target) as tgt:
        for i, (s, t) in bar(enumerate(zip(src, tgt))):
            writer.add_document(ID=str(i), SRC=s.strip(), TGT=t.strip())
    logger.info('Start committing...')
    writer.commit()
    logger.info('Finished committing')
    return ix


def open_index(index_path: Union[Path, str]) -> Index:
    if isinstance(index_path, str):
        index_path = Path(index_path)
    assert index_path.exists()
    return open_dir(index_path.absolute(), readonly=True)
