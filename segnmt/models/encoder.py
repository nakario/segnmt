from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

from segnmt.misc.constants import PAD
from segnmt.misc.typing import ndarray


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

    def __call__(self, source: ndarray) -> Variable:
        minibatch_size, max_sentence_size = source.shape

        embedded_source = self.embed_id(source)
        assert embedded_source.shape == \
            (minibatch_size, max_sentence_size, self.word_embeddings_size)

        embedded_sentences = F.separate(embedded_source, axis=0)
        sentence_masks: List[ndarray] = \
            self.xp.vsplit(source != PAD, minibatch_size)

        masked_sentences: List[Variable] = []
        for sentence, mask in zip(embedded_sentences, sentence_masks):
            masked_sentences.append(sentence[mask.reshape((-1,))])

        encoded_sentences: List[Variable] = \
            self.nstep_birnn(None, None, masked_sentences)[-1]
        assert len(encoded_sentences) == minibatch_size

        encoded: Variable = F.pad_sequence(
            encoded_sentences,
            length=max_sentence_size,
            padding=0.0
        )
        assert encoded.shape == \
            (minibatch_size, max_sentence_size, self.output_size)

        return encoded
