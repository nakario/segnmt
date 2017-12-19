from typing import Callable

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable


class AttentionModule(chainer.Chain):
    def __init__(
            self,
            encoder_output_size: int,
            attention_layer_size: int,
            decoder_hidden_layer_size: int,
            output_embedding_size: int
    ):
        super(AttentionModule, self).__init__()
        with self.init_scope():
            self.linear_h = L.Linear(encoder_output_size,
                                     attention_layer_size)
            self.linear_s = L.Linear(decoder_hidden_layer_size,
                                     attention_layer_size,
                                     nobias=True)
            self.linear_o = L.Linear(attention_layer_size,
                                     1, nobias=True)
            self.linear_e = L.Linear(output_embedding_size,
                                     attention_layer_size)
        self.encoder_output_size = encoder_output_size
        self.attention_layer_size = attention_layer_size
        self.precomputed = False
        self.encoded = None
        self.precomputed_alignment_factor = None

    def precompute(
            self,
            encoded: Variable
    ):
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size

        self.encoded = encoded
        self.precomputed_alignment_factor = F.reshape(
            self.linear_h(
                F.reshape(
                    encoded,
                    (minibatch_size * max_sentence_size, encoder_output_size)
                )
            ),
            (minibatch_size, max_sentence_size, self.attention_layer_size)
        )
        self.precomputed = True

    def __call__(
            self,
            previous_state: Variable,
            previous_embedding: Variable
    ) -> Variable:
        minibatch_size, max_sentence_size, _ = self.encoded.shape
        assert self.precomputed

        state_alignment_factor = \
            self.linear_s(previous_state) + \
            self.linear_e(previous_embedding)
        assert state_alignment_factor.shape == \
            (minibatch_size, self.attention_layer_size)

        attention = F.softmax(F.reshape(
            self.linear_o(
                F.reshape(
                    F.tanh(
                        self.precomputed_alignment_factor + F.broadcast_to(
                            F.expand_dims(state_alignment_factor, axis=1),
                            (
                                minibatch_size,
                                max_sentence_size,
                                self.attention_layer_size
                            )
                        )
                    ),
                    (
                        minibatch_size * max_sentence_size,
                        self.attention_layer_size
                    )
                )
            ),
            (minibatch_size, max_sentence_size)
        ))
        assert attention.shape == (minibatch_size, max_sentence_size)

        context = F.sum(F.scale(self.encoded, attention, axis=0), axis=1)
        return context

