from typing import List
from typing import Optional
from typing import Tuple

import chainer
from chainer import Variable

from segnmt.misc.typing import ndarray
from segnmt.models.encoder import Encoder
from segnmt.models.decoder import Decoder


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
                 maxout_layer_size: int):
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
                               encoder_hidden_layer_size * 2,
                               maxout_layer_size)

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
        return loss

    def translate(self, sentences: ndarray) -> List[ndarray]:
        # sentences.shape == (sentence_count, max_sentence_size)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            encoded = self.enc(sentences)
            translated = self.dec.translate(encoded)
            return translated

    def generate_context_memory(
            self,
            pairs: List[Tuple[ndarray, ndarray]],
    ) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
        # len(pairs) == max_retrieved_count
        contexts = []
        states = []
        logits = []
        betas = []
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for (source, target) in pairs:
                minibatch_size, max_sentence_size = source.shape
                assert target.shape[0] == minibatch_size
                encoded = self.enc(source)
                c, s, l = zip(*self.dec.generate_keys(encoded, target))
                contexts.extend(c)
                states.extend(s)
                logits.extend(l)
                betas.extend(
                    [self.xp.zeros((minibatch_size, 1), 'f')] * len(c)
                )
        contexts = self.xp.dstack(contexts)
        states = self.xp.dstack(states)
        logits = self.xp.dstack(logits)
        betas = self.xp.hstack(betas)
        return contexts, states, logits, betas
