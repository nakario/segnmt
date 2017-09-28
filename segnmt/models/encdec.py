from logging import getLogger
from typing import List

import chainer
from chainer import Variable

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

    def __call__(self, src_batch, src_mask, tgt_batch):
        encoded = self.enc(src_batch, src_mask)
        loss = self.dec(encoded, src_mask, tgt_batch)
        chainer.report({'loss': loss.data}, self)
        return loss

    def translate(self,
                  sentences: List[Variable],
                  masks: List[Variable]) -> List[Variable]:
        encoded = self.enc(sentences, masks)
        translated = self.dec.translate(encoded, masks)
        return translated
