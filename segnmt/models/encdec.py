from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

import chainer
from chainer import Variable

from segnmt.misc.typing import ndarray
from segnmt.models.encoder import Encoder
from segnmt.models.decoder import Decoder


logger = getLogger(__name__)


class EncoderDecoder(chainer.Chain):
    def __init__(self,
                 input_vocabulary_size: int,
                 input_word_embeddings_size: int,
                 encoder_hidden_layer_size: int,
                 encoder_num_steps: int,
                 encoder_dropout: float,
                 output_vocabulary_size: int,
                 output_word_embeddings_size: int,
                 decoder_hidden_layer_size: int,
                 attention_hidden_layer_size: int,
                 gate_hidden_layer_size: int,
                 maxout_layer_size: int,
                 fusion_mode: str):
        super(EncoderDecoder, self).__init__()
        with self.init_scope():
            self.enc = Encoder(input_vocabulary_size,
                               input_word_embeddings_size,
                               encoder_hidden_layer_size,
                               encoder_num_steps,
                               encoder_dropout)
            self.dec = Decoder(output_vocabulary_size,
                               output_word_embeddings_size,
                               decoder_hidden_layer_size,
                               attention_hidden_layer_size,
                               gate_hidden_layer_size,
                               encoder_hidden_layer_size * 2,
                               maxout_layer_size,
                               mode=fusion_mode)

    def __call__(
            self,
            source: ndarray,
            target: ndarray,
            similar_sentences: Optional[
                List[Tuple[ndarray, ndarray]]
            ] = None
    ) -> Variable:
        # source.shape == (minibatch_size, source_max_sentence_size)
        # target.shape == (minibatch_size, target_max_sentence_size)
        # len(similar_sentences) == max_retrieved_count
        encoded = self.enc(source)
        context_memory = None
        if similar_sentences is not None:
            context_memory = self.generate_context_memory(similar_sentences)
        loss = self.dec(encoded, target, context_memory)
        chainer.report({'loss': loss}, self)
        if similar_sentences is not None:
            chainer.report({'lambda': self.dec.E.l}, self)
            chainer.report({'gate': self.dec.averaged_gate}, self)
            chainer.report({'beta': self.dec.averaged_beta}, self)
            chainer.report({'max_score': Variable(self.dec.max_score)}, self)
        return loss

    def translate(
            self,
            sentences: ndarray,
            similar_sentences: Optional[
                List[Tuple[ndarray, ndarray]]
            ],
            max_translation_length: int = 100
    ) -> List[ndarray]:
        # sentences.shape == (sentence_count, max_sentence_size)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            encoded = self.enc(sentences)
            context_memory = None
            if similar_sentences is not None:
                context_memory = \
                    self.generate_context_memory(similar_sentences)
            translated = self.dec.translate(
                encoded,
                max_translation_length,
                context_memory
            )
            return translated

    def generate_context_memory(
            self,
            pairs: List[Tuple[ndarray, ndarray]],
    ) -> Tuple[ndarray, ndarray, ndarray]:
        # len(pairs) == max_retrieved_count
        contexts = []
        states = []
        indices = []
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for (source, target) in pairs:
                minibatch_size, max_sentence_size = source.shape
                assert target.shape[0] == minibatch_size
                encoded = self.enc(source)
                c, s, i = zip(*self.dec.generate_keys(encoded, target))
                contexts.extend(c)
                states.extend(s)
                indices.extend(i)
        contexts = self.xp.dstack(contexts).swapaxes(1,2)
        states = self.xp.dstack(states).swapaxes(1,2)
        indices = self.xp.vstack(indices).swapaxes(0,1)
        return contexts, states, indices
