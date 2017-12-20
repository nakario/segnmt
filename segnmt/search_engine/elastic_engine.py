from logging import getLogger
from logging import WARNING
from pathlib import Path
from typing import List
from typing import Tuple

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from elasticsearch_dsl import Mapping
from elasticsearch_dsl import Text

from segnmt.search_engine.retriever import BaseEngine


logger = getLogger(__name__)
es = Elasticsearch()
getLogger('elasticsearch').setLevel(WARNING)


class ElasticEngine(BaseEngine):
    def __init__(self, limit, index, doc_type):
        self.limit = limit
        self.index = index
        self.doc_type = doc_type

    def search(self, sentence: str) -> List[Tuple[str, str, str]]:
        body = {"query": {"match": {"SRC": sentence}}}
        results = es.search(
            index=self.index,
            doc_type=self.doc_type,
            body=body,
            size=self.limit,
            request_timeout=60
        )
        return [
            (r['_id'], r['_source']['SRC'], r['_source']['TGT'])
            for r in results['hits']['hits']
        ]


def create_index(index: str, doc_type: str, source: Path, target: Path):
    assert source.exists()
    assert target.exists()

    if es.indices.exists(index):
        es.indices.delete(index)
    m = Mapping(doc_type)
    m.field('SRC', Text(analyzer='whitespace'))
    m.field('TGT', Text(analyzer='whitespace'))
    m.save(index, using=es)

    actions = []
    with open(source) as src, open(target) as tgt:
        for i, (s, t) in enumerate(zip(src, tgt)):
            actions.append({
                "_index": index,
                "_type": doc_type,
                "_id": i,
                "_source": {
                    "SRC": s.strip(),
                    "TGT": t.strip()
                }
            })

    logger.info('Start query')
    bulk(es, actions, index=index, doc_type=doc_type)

    logger.info('Finish creating index')
