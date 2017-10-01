import argparse
from logging import getLogger
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple
from typing import Union

import chainer
from chainer.dataset import to_device
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import Variable
import matplotlib
import numpy as np
from progressbar import ProgressBar

from segnmt.misc.constants import EOS
from segnmt.misc.constants import PAD
from segnmt.misc.constants import UNK
from segnmt.misc.typing import ndarray
from segnmt.models.encdec import EncoderDecoder


logger = getLogger(__name__)
matplotlib.use('Agg')


class ConstArguments(NamedTuple):
    # Encoder-Decoder arguments
    source_vocabulary_size: int
    source_word_embeddings_size: int
    encoder_hidden_layer_size: int
    encoder_num_steps: int
    encoder_dropout: float
    target_vocabulary_size: int
    target_word_embeddings_size: int
    decoder_hidden_layer_size: int
    attention_hidden_layer_size: int
    maxout_layer_size: int

    gpu: int
    minibatch_size: int
    epoch: int
    source_vocab: str
    target_vocab: str
    training_source: str
    training_target: str
    validation_source: Optional[str]
    validation_target: Optional[str]
    loss_plot_file: str
    resume_file: Optional[str]
    min_source_len: int
    max_source_len: int
    min_target_len: int
    max_target_len: int

    run: Callable[[argparse.Namespace], None]


def convert(
        minibatch: List[Tuple[np.ndarray, np.ndarray]],
        device: Optional[int]
) -> Tuple[ndarray, ndarray]:
    # Append eos to the end of sentence
    eos = np.array([EOS], 'i')
    src_batch, tgt_batch = zip(*minibatch)
    with chainer.no_backprop_mode():
        src_sentences = \
            [Variable(np.hstack((sentence, eos))) for sentence in src_batch]
        tgt_sentences = \
            [Variable(np.hstack((sentence, eos))) for sentence in tgt_batch]

        src_block = F.pad_sequence(src_sentences, padding=PAD).data
        tgt_block = F.pad_sequence(tgt_sentences, padding=PAD).data

    return (
        to_device(device, src_block),
        to_device(device, tgt_block)
    )


def load_vocab(vocab_file: Union[Path, str], size: int) -> Dict[str, int]:
    """Create a vocabulary from a file.

    The file specified by `vocab` must be contain one word per line.
    """

    if isinstance(vocab_file, str):
        vocab_file = Path(vocab_file)
    assert vocab_file.exists()

    words = ['<UNK>', '<EOS>']
    with open(vocab_file) as f:
        words += [line.strip() for line in f]
    assert size <= len(words)

    vocab = {word: index for index, word in enumerate(words) if index < size}
    assert vocab['<UNK>'] == UNK
    assert vocab['<EOS>'] == EOS

    return vocab


def load_data(
        source: Union[Path, str],
        target: Union[Path, str],
        source_vocab: Dict[str, int],
        target_vocab: Dict[str, int],
        min_src_len: int,
        max_src_len: int,
        min_tgt_len: int,
        max_tgt_len: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if isinstance(source, str):
        source = Path(source)
    if isinstance(target, str):
        target = Path(target)
    assert source.exists()
    assert target.exists()

    data = []

    with open(source) as src, open(target) as tgt:
        src_len = sum(1 for _ in src)
        tgt_len = sum(1 for _ in tgt)
        assert src_len == tgt_len
        file_len = src_len

    logger.info(f'loading {source.absolute()} and {target.absolute()}')
    with open(source) as src, open(target) as tgt:
        bar = ProgressBar()
        for s, t in bar(zip(src, tgt), max_value=file_len):
            s_words = s.strip().split()
            t_words = t.strip().split()
            if len(s_words) < min_src_len or max_src_len < len(s_words):
                continue
            if len(t_words) < min_tgt_len or max_tgt_len < len(t_words):
                continue
            s_array = \
                np.array([source_vocab.get(w, UNK) for w in s_words], 'i')
            t_array = \
                np.array([target_vocab.get(w, UNK) for w in t_words], 'i')
            data.append((s_array, t_array))

    return data


def train(args: argparse.Namespace):
    cargs = ConstArguments(**vars(args))
    model = EncoderDecoder(cargs.source_vocabulary_size,
                           cargs.source_word_embeddings_size,
                           cargs.encoder_hidden_layer_size,
                           cargs.encoder_num_steps,
                           cargs.encoder_dropout,
                           cargs.target_vocabulary_size,
                           cargs.target_word_embeddings_size,
                           cargs.decoder_hidden_layer_size,
                           cargs.attention_hidden_layer_size,
                           cargs.maxout_layer_size)
    if cargs.gpu >= 0:
        chainer.cuda.get_device_from_id(cargs.gpu).use()
        model.to_gpu(cargs.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    source_vocab = load_vocab(cargs.source_vocab, cargs.source_vocabulary_size)
    target_vocab = load_vocab(cargs.target_vocab, cargs.target_vocabulary_size)

    training_data = load_data(
        cargs.training_source,
        cargs.training_target,
        source_vocab,
        target_vocab,
        cargs.min_source_len,
        cargs.max_source_len,
        cargs.min_target_len,
        cargs.max_target_len
    )

    training_iter = chainer.iterators.SerialIterator(training_data,
                                                     cargs.minibatch_size)
    updater = training.StandardUpdater(
        training_iter, optimizer, converter=convert, device=cargs.gpu)
    trainer = training.Trainer(updater, (cargs.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(10, 'iteration')))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'elapsed_time']
        ),
        trigger=(10, 'iteration')
    )
    trainer.extend(extensions.snapshot(), trigger=(50, 'iteration'))
    trainer.extend(extensions.dump_graph('main/loss'))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'],
            'epoch',
            file_name=cargs.loss_plot_file
        ))
    else:
        logger.warning('PlotReport is not available.')

    if cargs.validation_source is not None and \
            cargs.validation_target is not None:
        validation_data = load_data(
            cargs.validation_source,
            cargs.validation_target,
            source_vocab,
            target_vocab,
            cargs.min_source_len,
            cargs.max_source_len,
            cargs.min_target_len,
            cargs.max_target_len
        )

        validation_size = len(validation_data)

        source_word = {index: word for word, index in source_vocab.items()}
        target_word = {index: word for word, index in target_vocab.items()}

        logger.info(f'Validation data: {validation_size}')

        @chainer.training.make_extension(trigger=(50, 'iteration'))
        def translate(_):
            data = validation_data[np.random.choice(validation_size)]
            source, target = convert([data], cargs.gpu)
            result = model.translate(source)[0].reshape((1, -1))

            source_sentence = ' '.join(
                [source_word[int(word)] for word in source[0]]
            )
            target_sentence = ' '.join(
                [target_word[int(word)] for word in target[0]]
            )
            result_sentence = ' '.join(
                [target_word[int(word)] for word in result[0]]
            )
            logger.info('# source : ' + source_sentence)
            logger.info('# result : ' + result_sentence)
            logger.info('# expect : ' + target_sentence)

        trainer.extend(translate, trigger=(50, 'iteration'))

    trainer.extend(extensions.ProgressBar(update_interval=1))

    if cargs.resume_file is not None:
        chainer.serializers.load_npz(cargs.resume_file, trainer)

    print('start training')

    trainer.run()
