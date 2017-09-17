import chainer
import chainer.links as L

from segnmt.models.attention import AttentionModule


class Decoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 attention_hidden_layer_size: int,
                 encoder_output_size: int,
                 maxout_layer_size: int,
                 maxout_pool_size: int = 2):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(vocabulary_size, word_embeddings_size)
            self.rnn = L.LSTM(word_embeddings_size + encoder_output_size,
                              hidden_layer_size)
            self.maxout = L.Maxout(word_embeddings_size +
                                   encoder_output_size +
                                   hidden_layer_size,
                                   maxout_layer_size,
                                   maxout_pool_size)
            self.linear = L.Linear(maxout_layer_size, vocabulary_size)
            self.attention = AttentionModule(encoder_output_size,
                                             attention_hidden_layer_size,
                                             hidden_layer_size)
