from logging import getLogger
from pathlib import Path
from typing import List
from typing import Tuple
from typing import Union

from progressbar import ProgressBar
from whoosh.fields import ID
from whoosh.fields import Schema
from whoosh.fields import TEXT
from whoosh.index import Index
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.qparser import OrGroup
from whoosh.qparser import QueryParser

from segnmt.search_engine.retriever import BaseEngine


logger = getLogger(__name__)


class WhooshEngine(BaseEngine):
    def __init__(self, limit, index):
        self.limit = limit
        self.index = index
        self.parser = QueryParser("SRC", index.schema, group=OrGroup)

    def search(self, sentence: str) -> List[Tuple[str, str]]:
        with self.index.searcher() as searcher:
            query = self.parser.parse(sentence)
            results = searcher.search(query, limit=self.limit)
            return [(r["SRC"], r["TGT"]) for r in results]


def create_index(index_path: Union[Path, str],
                 source: List[str],
                 target: List[str]) -> Index:
    if isinstance(index_path, str):
        index_path = Path(index_path)
    assert index_path.exists()
    assert len(source) == len(target)
    schema = Schema(ID=ID, SRC=TEXT(stored=True), TGT=TEXT)
    ix = create_in(index_path.absolute(), schema)
    writer = ix.writer()
    bar = ProgressBar(max_value=len(source))
    logger.info(f'Creating index at {index_path.absolute()}')
    for i, (src, tgt) in bar(enumerate(zip(source, target))):
        writer.add_document(ID=str(i), SRC=src.strip(), TGT=tgt.strip())
    writer.commit()
    return ix


def open_index(index_path: Union[Path, str]) -> Index:
    if isinstance(index_path, str):
        index_path = Path(index_path)
    assert index_path.exists()
    return open_dir(index_path.absolute(), readonly=True)
