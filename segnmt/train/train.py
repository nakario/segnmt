from logging import getLogger
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import chainer
from chainer.dataset import to_device
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer import Variable
import numpy as np
from progressbar import ProgressBar

from segnmt.commands.train import ConstArguments
from segnmt.misc.constants import EOS
from segnmt.misc.constants import PAD
from segnmt.misc.constants import UNK
from segnmt.models.encdec import EncoderDecoder


logger = getLogger(__name__)


def convert(
        minibatch: List[Tuple[np.ndarray, np.ndarray]],
        device: Optional[int]
) -> Tuple[List[Variable], List[Variable], List[Variable]]:
    # Append eos to the end of sentence
    eos = np.array([EOS], 'i')
    src_batch, tgt_batch = zip(*minibatch)
    src_sentences = \
        [Variable(to_device(device, np.hstack((s, eos)))) for s in src_batch]
    tgt_sentences = \
        [Variable(to_device(device, np.hstack((t, eos)))) for t in tgt_batch]

    src_block = F.pad_sequence(src_sentences, padding=PAD)
    tgt_block = F.pad_sequence(tgt_sentences, padding=PAD)
    mask_block = Variable(src_block.data != PAD)

    return (
        F.separate(src_block, axis=1),
        F.separate(mask_block, axis=1),
        F.separate(tgt_block, axis=1)
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

    vocab = {word: index for index, word in enumerate(words)}
    assert vocab['<UNK>'] == UNK
    assert vocab['<EOS>'] == EOS

    return vocab[:size]


def load_data(
        source: Union[Path, str],
        target: Union[Path, str],
        source_vocab: Dict[str, int],
        target_vocab: Dict[str, int],
        min_len: int,
        max_len: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if isinstance(source, str):
        source = Path(source)
    if isinstance(target, str):
        target = Path(target)
    assert source.exists()
    assert target.exists()

    data = []

    logger.info(f'loading {source.absolute()} and {target.absolute()}')
    with open(source) as src, open(target) as tgt:
        src_len = sum(1 for _ in src)
        tgt_len = sum(1 for _ in tgt)
        assert src_len == tgt_len

        bar = ProgressBar()
        for s, t in bar(zip(src, tgt), max_value=src_len):
            s_words = s.strip().split()
            t_words = t.strip().split()
            if len(s_words) < min_len or max_len < len(s_words):
                continue
            if len(t_words) < min_len or max_len < len(t_words):
                continue
            s_array = \
                np.array([source_vocab.get(w, UNK) for w in s_words], 'i')
            t_array = \
                np.array([target_vocab.get(w, UNK) for w in t_words], 'i')
            data.append((s_array, t_array))

    return data


def train(cargs: ConstArguments):
    model = EncoderDecoder(cargs.input_vocabulary_size,
                           cargs.input_word_embeddings_size,
                           cargs.encoder_hidden_layer_size,
                           cargs.encoder_num_steps,
                           cargs.encoder_dropout,
                           cargs.output_vocabulary_size,
                           cargs.output_word_embeddings_size,
                           cargs.decoder_hidden_layer_size,
                           cargs.attention_hidden_layer_size,
                           cargs.maxout_layer_size)
    if cargs.gpu >= 0:
        chainer.cuda.get_device_from_id(cargs.gpu).use()
        model.to_gpu(cargs.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    source_vocab = load_vocab(cargs.source_vocab, cargs.source_vocab_size)
    target_vocab = load_vocab(cargs.target_vocab, cargs.target_vocab_size)

    training_data = load_data(
        cargs.training_source,
        cargs.training_target,
        source_vocab,
        target_vocab,
        cargs.min_source_len,
        cargs.max_source_len
    )

    training_iter = chainer.iterators.SerialIterator(training_data,
                                                     cargs.minibatch_size)
    updater = training.StandardUpdater(
        training_iter, optimizer, converter=convert, device=cargs.gpu)
    trainer = training.Trainer(updater, (cargs.epoch, 'epoch'))
    trainer.extend(extensions.LogReport(trigger=(200, 'iteration')))
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'elapsed_time']
        ),
        trigger=(200, 'iteration')
    )

    if cargs.validation_source is not None and \
            cargs.validation_target is not None:
        validation_data = load_data(
            cargs.validation_source,
            cargs.validation_target,
            source_vocab,
            target_vocab,
            cargs.min_source_len,
            cargs.max_source_len
        )

        validation_size = len(validation_data)

        source_word = {index: word for word, index in source_vocab.items()}
        target_word = {index: word for word, index in target_vocab.items()}

        logger.info(f'Validation data: {validation_size}')

        @chainer.training.make_extension(trigger=(200, 'iteration'))
        def translate(_):
            data = validation_data[model.xp.random.choice(validation_size)]
            source, mask, target = convert([data], cargs.gpu)
            result = F.separate(model.translate(source, mask)[0], axis=0)

            source_sentence = ' '.join([source_word[x] for x in source])
            target_sentence = ' '.join([target_word[y] for y in target])
            result_sentence = ' '.join([target_word[y] for y in result])
            logger.info('# source : ' + source_sentence)
            logger.info('# result : ' + result_sentence)
            logger.info('# expect : ' + target_sentence)

        trainer.extend(translate, trigger=(4000, 'iteration'))

    print('start training')

    trainer.run()
