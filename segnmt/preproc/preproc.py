from argparse import Namespace
from collections import Counter
from logging import getLogger
from pathlib import Path
from typing import List
from typing import NamedTuple
from typing import Union

from joblib import delayed
from joblib import Parallel

from segnmt.search_engine.retriever import Retriever
from segnmt.search_engine.similarity import fuzzy_word_level_similarity
from segnmt.search_engine.elastic_engine import ElasticEngine
from segnmt.search_engine.elastic_engine import create_index


class ConstArguments(NamedTuple):
    source: str
    target: str
    output: str
    max_source_len: int
    max_target_len: int
    source_dev: str
    target_dev: str
    use_existing_files: bool


logger = getLogger(__name__)


def copy_data_with_limit(
        source: Union[str, Path],
        target: Union[str, Path],
        source_copy: Union[str, Path],
        target_copy: Union[str, Path],
        min_source_len: int,
        max_source_len: int,
        min_target_len: int,
        max_target_len: int
):
    if isinstance(source, str):
        source = Path(source)
    if isinstance(target, str):
        target = Path(target)
    if isinstance(source_copy, str):
        source_copy = Path(source_copy)
    if isinstance(target_copy, str):
        target_copy = Path(target_copy)
    assert source.exists()
    assert target.exists()
    assert source_copy.parent.exists()
    assert target_copy.parent.exists()

    with open(source) as src,\
            open(target) as tgt,\
            open(source_copy, 'w') as src_c,\
            open(target_copy, 'w') as tgt_c:
        for s, t in zip(src, tgt):
            s_words = s.strip().split()
            t_words = t.strip().split()
            if len(s_words) < min_source_len or max_source_len < len(s_words):
                continue
            if len(t_words) < min_target_len or max_target_len < len(t_words):
                continue
            src_c.write(s)
            tgt_c.write(t)


def make_voc(
        document: Union[str, Path],
        out_file: Union[str, Path],
):
    """Create a vocabulary file."""
    if isinstance(document, str):
        document = Path(document)
    if isinstance(out_file, str):
        out_file = Path(out_file)
    assert document.exists()
    assert out_file.parent.exists()

    logger.info(f'Preprocessing {document.absolute()}')
    sentence_count = 0
    word_count = 0
    counts = Counter()
    with open(document) as doc:
        for sentence in doc:
            sentence_count += 1
            words = sentence.strip().split()
            word_count += len(words)
            for word in words:
                counts[word] += 1

    vocab = [word for (word, _) in counts.most_common()]
    with open(out_file, 'w') as out:
        for word in vocab:
            out.write(word)
            out.write('\n')

    logger.info(f'Number of sentences: {sentence_count}')
    logger.info(f'Number of words    : {word_count}')
    logger.info(f'Size of vocabulary : {len(vocab)}')


def retrieve_indices(sentence: str, i: int, training: bool) -> List[str]:
    engine = ElasticEngine(100, 'segnmt', 'pairs')
    retriever = Retriever(
        engine,
        fuzzy_word_level_similarity,
        training=training
    )
    retrieved = retriever.retrieve(sentence, i)
    if len(retrieved) > 0:
        retrieved_indices, _, _ = zip(*retrieved)
        return [str(i)] + list(retrieved_indices)
    else:
        return [str(i)]


def make_sim(
        data: Union[Path, str],
        sim_file: Union[Path, str],
        training: bool
):
    """Create a list of indices of similar sentences."""
    if isinstance(data, str):
        data = Path(data)
    if isinstance(sim_file, str):
        sim_file = Path(sim_file)
    assert data.exists()
    assert sim_file.parent.exists()

    indices_list: List[List[str]]
    with open(data) as src:
        sentence_list = src.readlines()
        indices_list = Parallel(n_jobs=-1, verbose=1)([
            delayed(retrieve_indices)(sentence.strip(), i, training)
            for i, sentence in enumerate(sentence_list)
        ])

    with open(sim_file, 'w') as sim:
        for indices in indices_list:
            sim.write(' '.join(indices) + '\n')


def make_config(config_file: Path):
    pass


def preproc(args: Namespace):
    cargs = ConstArguments(**vars(args))
    source = Path(cargs.output) / Path('source')
    target = Path(cargs.output) / Path('target')
    output = Path(cargs.output)
    if not output.exists():
        logger.warning(f'{output.absolute()} does not exist')
        output.mkdir(parents=True, exist_ok=True)
    if not cargs.use_existing_files:
        copy_data_with_limit(
            cargs.source, cargs.target,
            source, target,
            1, cargs.max_source_len,
            1, cargs.max_target_len
        )
        make_voc(source, output / Path('source_voc'))
        make_voc(target, output / Path('target_voc'))
        create_index('segnmt', 'pairs', source, target)
    make_sim(
        source,
        output / Path('train_sim'),
        True
    )
    if cargs.source_dev is None or cargs.target_dev is None:
        return
    source_dev = output / Path('source_dev')
    target_dev = output / Path('target_dev')
    if not cargs.use_existing_files:
        copy_data_with_limit(
            cargs.source_dev, cargs.target_dev,
            source_dev, target_dev,
            1, cargs.max_source_len,
            1, cargs.max_target_len
        )
    make_sim(
        source_dev,
        output / Path('dev_sim'),
        False
    )
