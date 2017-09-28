from logging import getLogger
from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable


logger = getLogger(__name__)


class Encoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 num_steps: int,
                 dropout: float,
                 ignore_label: int = -1):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.embed_id = L.EmbedID(vocabulary_size,
                                      word_embeddings_size,
                                      ignore_label=ignore_label)
            self.nstep_birnn = L.NStepBiLSTM(num_steps,
                                             word_embeddings_size,
                                             hidden_layer_size,
                                             dropout)
        self.word_embeddings_size = word_embeddings_size
        self.output_size = hidden_layer_size * 2

    def __call__(self,
                 sequence: List[Variable],
                 mask: List[Variable]) -> Variable:
        assert len(sequence) == len(mask)

        minibatch_size = sequence[0].shape[0]
        max_sentence_size = len(sequence)
        word_embeddings_size = self.word_embeddings_size
        output_size = self.output_size

        sentence_matrix = F.transpose(F.vstack(sequence))
        assert sentence_matrix.shape == (minibatch_size, max_sentence_size)

        mask_matrix = F.transpose(F.vstack(mask))
        assert mask_matrix.shape == (minibatch_size, max_sentence_size)

        embedded_sentence_matrix = self.embed_id(sentence_matrix)
        assert embedded_sentence_matrix.shape == \
            (minibatch_size, max_sentence_size, word_embeddings_size)

        embedded_sentences: List[Variable] = \
            F.separate(embedded_sentence_matrix, axis=0)
        sentence_masks: List[Variable] = F.separate(mask_matrix, axis=0)

        masked_sentences: List[Variable] = []
        for sentence, mask_ in zip(embedded_sentences, sentence_masks):
            masked_sentences.append(sentence[mask_.data])
        for i, ms in enumerate(masked_sentences):
            logger.debug(f'masked:  {ms.shape}')

        encoded_sentences: List[Variable] = \
            self.nstep_birnn(None, None, masked_sentences)[-1]
        for i, es in enumerate(encoded_sentences):
            logger.debug(f'encoded: {es.shape}')
        assert len(encoded_sentences) == minibatch_size
        logger.debug(f'encoded: {encoded_sentences}')

        encoded_sentence_matrix = F.pad_sequence(encoded_sentences, padding=0)
        logger.debug(f'matrix: {encoded_sentence_matrix.shape}')
        assert encoded_sentence_matrix.shape == \
            (minibatch_size, max_sentence_size, output_size)

        return encoded_sentence_matrix
