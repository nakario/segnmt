import chainer
import chainer.links as L


class Encoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 num_steps: int,
                 dropout: float):
        super(Encoder, self).__init__()
        with self.init_scope():
            self.emb = L.EmbedID(vocabulary_size, word_embeddings_size)
            self.nstep_birnn = L.NStepBiGRU(num_steps,
                                            word_embeddings_size,
                                            hidden_layer_size,
                                            dropout)
