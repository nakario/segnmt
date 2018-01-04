from argparse import Namespace
from collections import Counter
from logging import getLogger
from pathlib import Path
from time import sleep
from typing import Callable
from typing import List
from typing import NamedTuple
from typing import Union

from joblib import delayed
from joblib import Parallel

from segnmt.external_libs.bpe import learn_bpe
from segnmt.external_libs.bpe import apply_bpe
from segnmt.search_engine.retriever import Retriever
from segnmt.search_engine.similarity import functions
from segnmt.search_engine.elastic_engine import ElasticEngine
from segnmt.search_engine.elastic_engine import create_index


class ConstArguments(NamedTuple):
    source: str
    target: str
    output: str
    min_source_len: int
    max_source_len: int
    min_target_len: int
    max_target_len: int
    source_dev: str
    target_dev: str
    source_test: str
    target_test: str
    skip_create_index: bool
    skip_sleep: bool
    skip_make_sim: bool
    skip_create_bpe: bool
    skip_bpe_encode: bool
    skip_make_voc: bool
    limit: int
    similarity_function: str
    sleep_time: int


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


def create_bpe_file(
        document: Union[Path, str],
        bpe_file: Union[Path, str]
):
    if isinstance(document, str):
        document = Path(document)
    if isinstance(bpe_file, str):
        bpe_file = Path(bpe_file)
    assert document.exists()
    assert bpe_file.parent.exists()

    logger.info('Start learning BPE')
    with open(document) as doc, open(bpe_file, 'w') as out:
        iterable = map(lambda x: x.strip().split(), doc)
        learn_bpe.learn_bpe_from_sentence_iterable(
            iterable,
            out,
            symbols=10000,
            min_frequency=2,
            verbose=False
        )


def bpe_encode(
        document: Union[Path, str],
        compressed: Union[Path, str],
        bpe_file: Union[Path, str]
):
    if isinstance(document, str):
        document = Path(document)
    if isinstance(compressed, str):
        compressed = Path(compressed)
    if isinstance(bpe_file, str):
        bpe_file = Path(bpe_file)
    assert document.exists()
    assert compressed.parent.exists()
    assert bpe_file.exists()

    with open(bpe_file) as file:
        bpe = apply_bpe.BPE(file, separator='._@@@')
    logger.info(f'Start applying BPE to {document.absolute()}')
    with open(document) as src, open(compressed, 'w') as src_bpe:
        for line in src:
            src_bpe.write(
                ' '.join(bpe.segment_splitted(line.strip().split())) + '\n'
            )


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


def retrieve_indices(
        sentence: str,
        i: int,
        training: bool,
        similar_function: Callable[[str, str], float],
        limit: int = -1
) -> List[str]:
    engine = ElasticEngine(100, 'segnmt', 'pairs')
    retriever = Retriever(
        engine,
        similar_function,
        limit=limit,
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
        training: bool,
        similar_function: Callable[[str, str], float],
        limit: int = -1
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
            delayed(retrieve_indices)(
                sentence.strip(), i, training, similar_function, limit
            )
            for i, sentence in enumerate(sentence_list)
        ])

    with open(sim_file, 'w') as sim:
        for indices in indices_list:
            sim.write(' '.join(indices) + '\n')


def make_config(config_file: Path):
    pass


def preproc(args: Namespace):
    cargs = ConstArguments(**vars(args))

    output = Path(cargs.output)
    if not output.exists():
        logger.warning(f'{output.absolute()} does not exist')
        output.mkdir(parents=True, exist_ok=True)
        logger.info(f'created {output.absolute()}')

    # training dataset
    source = output / Path('source')
    target = output / Path('target')

    copy_data_with_limit(
        cargs.source, cargs.target,
        source, target,
        cargs.min_source_len, cargs.max_source_len,
        cargs.min_target_len, cargs.max_target_len
    )
    if not cargs.skip_create_index:
        create_index('segnmt', 'pairs', source, target)
    if not cargs.skip_sleep:
        sleep(cargs.sleep_time)
    if not cargs.skip_make_sim:
        make_sim(
            source,
            output / Path('train_sim'),
            True,
            functions.get(cargs.similarity_function),
            cargs.limit
        )

    bpe_source = output / Path('bpe_source')
    bpe_target = output / Path('bpe_target')
    if not cargs.skip_create_bpe:
        create_bpe_file(source, bpe_source)
        create_bpe_file(target, bpe_target)
    source_compressed = output / Path('source_compressed')
    target_compressed = output / Path('target_compressed')
    if not cargs.skip_bpe_encode:
        bpe_encode(source, source_compressed, bpe_source)
        bpe_encode(target, target_compressed, bpe_target)
    if not cargs.skip_make_voc:
        make_voc(source_compressed, output / Path('source_bpe_voc'))
        make_voc(target_compressed, output / Path('target_bpe_voc'))

    # validation dataset
    if cargs.source_dev is None or cargs.target_dev is None:
        return
    source_dev = output / Path('source_dev')
    target_dev = output / Path('target_dev')
    copy_data_with_limit(
        cargs.source_dev, cargs.target_dev,
        source_dev, target_dev,
        cargs.min_source_len, cargs.max_source_len,
        cargs.min_target_len, cargs.max_target_len
    )
    if not cargs.skip_make_sim:
        make_sim(
            source_dev,
            output / Path('dev_sim'),
            False,
            functions.get(cargs.similarity_function),
            cargs.limit
        )

    source_dev_compressed = output / Path('source_dev_compressed')
    target_dev_compressed = output / Path('target_dev_compressed')
    if not cargs.skip_bpe_encode:
        bpe_encode(source_dev, source_dev_compressed, bpe_source)
        bpe_encode(target_dev, target_dev_compressed, bpe_target)

    # test dataset
    if cargs.source_test is None or cargs.target_test is None:
        return
    source_test = output / Path('source_test')
    target_test = output / Path('target_test')
    copy_data_with_limit(
        cargs.source_test, cargs.target_test,
        source_test, target_test,
        1, 1000,
        1, 1000
    )
    if not cargs.skip_make_sim:
        make_sim(
            source_test,
            output / Path('test_sim'),
            False,
            functions.get(cargs.similarity_function),
            cargs.limit
        )

    source_test_compressed = output / Path('source_test_compressed')
    target_test_compressed = output / Path('target_test_compressed')
    if not cargs.skip_bpe_encode:
        bpe_encode(source_test, source_test_compressed, bpe_source)
        bpe_encode(target_test, target_test_compressed, bpe_target)
