import chainer
import chainer.links as L


class AttentionModule(chainer.Chain):
    def __init__(self,
                 encoder_output_size: int,
                 attention_layer_size: int,
                 decoder_hidden_layer_size: int):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.linear_h = L.Linear(encoder_output_size,
                                     attention_layer_size)
            self.linear_s = L.Linear(decoder_hidden_layer_size,
                                     attention_layer_size,
                                     nobias=True)
            self.linear_o = L.Linear(attention_layer_size,
                                     1, nobias=True)
