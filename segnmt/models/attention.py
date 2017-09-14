import chainer
import chainer.links as L


class AttentionModule(chainer.Chain):
    def __init__(self,
                 input_hidden_layer_size: int,
                 attention_layer_size: int,
                 outpu_hidden_layer_size: int):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.linear_h = L.Linear(input_hidden_layer_size,
                                     attention_layer_size)
            self.linear_s = L.Linear(outpu_hidden_layer_size,
                                     attention_layer_size,
                                     nobias=True)
            self.linear_o = L.Linear(attention_layer_size,
                                     1, nobias=True)
