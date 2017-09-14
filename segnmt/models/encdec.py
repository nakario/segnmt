import chainer

from segnmt.models.encoders import Encoder
from segnmt.models.decoders import Decoder


class EncoderDecoder(chainer.Chain):
    def __init__(self,
                 input_vocabulary_size: int,
                 input_word_embeddings_size: int,
                 encoder_hidden_layer_size: int,
                 num_steps: int,
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
                               num_steps)
            self.dec = Decoder(output_vocabulary_size,
                               output_word_embeddings_size,
                               decoder_hidden_layer_size,
                               attention_hidden_layer_size,
                               encoder_hidden_layer_size * 2,
                               maxout_layer_size)
