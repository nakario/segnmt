import argparse
from logging import getLogger
from typing import List
from typing import NamedTuple
from typing import Optional

import chainer
import matplotlib
from nltk.translate import bleu_score

from segnmt.models.encdec import EncoderDecoder
from segnmt.train.train import decode_bpe
from segnmt.train.train import convert
from segnmt.train.train import convert_with_similar_sentences
from segnmt.train.train import load_vocab
from segnmt.train.train import load_data
from segnmt.train.train import load_validation_data


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
    gate_hidden_layer_size: int
    maxout_layer_size: int
    fusion_mode: str

    gpu: int
    minibatch_size: int
    source_vocab: str
    target_vocab: str
    training_source: str
    training_target: str
    validation_source: str
    validation_target: str
    similar_sentence_indices: Optional[str]
    similar_sentence_indices_validation: Optional[str]
    translation_output_file: str
    resume_file: str
    max_translation_length: int


def evaluate(args: argparse.Namespace):
    cargs = ConstArguments(**vars(args))
    logger.info(f'cargs: {cargs}')
    model = EncoderDecoder(cargs.source_vocabulary_size,
                           cargs.source_word_embeddings_size,
                           cargs.encoder_hidden_layer_size,
                           cargs.encoder_num_steps,
                           cargs.encoder_dropout,
                           cargs.target_vocabulary_size,
                           cargs.target_word_embeddings_size,
                           cargs.decoder_hidden_layer_size,
                           cargs.attention_hidden_layer_size,
                           cargs.gate_hidden_layer_size,
                           cargs.maxout_layer_size,
                           cargs.fusion_mode)
    if cargs.gpu >= 0:
        chainer.cuda.get_device_from_id(cargs.gpu).use()
        model.to_gpu(cargs.gpu)

    chainer.serializers.load_npz(cargs.resume_file, model)

    source_vocab = load_vocab(cargs.source_vocab, cargs.source_vocabulary_size)
    target_vocab = load_vocab(cargs.target_vocab, cargs.target_vocabulary_size)

    converter = convert
    if cargs.similar_sentence_indices is not None:
        converter = convert_with_similar_sentences

    if cargs.similar_sentence_indices_validation is not None:
        validation_data = load_validation_data(
            cargs.training_source,
            cargs.training_target,
            cargs.validation_source,
            cargs.validation_target,
            source_vocab,
            target_vocab,
            cargs.similar_sentence_indices_validation,
            1000
        )
    else:
        validation_data = load_data(
            cargs.validation_source,
            cargs.validation_target,
            source_vocab,
            target_vocab
        )

    v_iter = chainer.iterators.SerialIterator(
        validation_data,
        cargs.minibatch_size,
        repeat=False,
        shuffle=False
    )

    target_sentences: List[List[List[str]]]
    with open(cargs.validation_target) as f:
        target_sentences = \
            list(map(lambda x: [x.strip().split()], f.readlines()))

    target_word = {index: word for word, index in target_vocab.items()}

    list_of_references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []
    v_iter.reset()
    print("start translation")
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        i = 0
        for minibatch in v_iter:
            list_of_references.extend(target_sentences[i:i+len(minibatch)])
            i = i + len(minibatch)
            converted = converter(minibatch, cargs.gpu)
            source = converted[0]
            similars = None
            if len(converted) == 3:
                similars = converted[2]
            results = model.translate(
                source,
                similars,
                max_translation_length=cargs.max_translation_length
            )
            hypotheses.extend([
                decode_bpe([
                    target_word.get(id_, '<UNK>')
                    for id_ in sentence.tolist()[:-1]
                ]) for sentence in results
            ])
    print("start write file")
    assert len(list_of_references) == len(hypotheses)
    with open(cargs.translation_output_file, 'w') as output:
        for i in range(len(list_of_references)):
            output.write(f"src: {' '.join(list_of_references[i][0])}\n")
            output.write(f"out: {' '.join(hypotheses[i])}\n\n")
    print("start calc bleu")
    bleu = bleu_score.corpus_bleu(
        list_of_references,
        hypotheses
    )
    print(f"BLEU: {bleu}")
