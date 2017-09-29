from typing import Callable

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable


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
        self.encoder_output_size = encoder_output_size
        self.attention_layer_size = attention_layer_size

    def __call__(self,
                 encoded: Variable) -> Callable[[Variable], Variable]:
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size

        precomputed_alignment_factor = F.reshape(
            self.linear_h(
                F.reshape(
                    encoded,
                    (minibatch_size * max_sentence_size, encoder_output_size)
                )
            ),
            (minibatch_size, max_sentence_size, self.attention_layer_size)
        )

        def compute_context(previous_state: Variable) -> Variable:
            state_alignment_factor = self.linear_s(previous_state)
            assert state_alignment_factor.shape == \
                (minibatch_size, self.attention_layer_size)
            state_alignment_factor_broadcast = F.broadcast_to(
                F.reshape(
                    state_alignment_factor,
                    (minibatch_size, 1, self.attention_layer_size)
                ),
                (minibatch_size, max_sentence_size, self.attention_layer_size)
            )
            scores = F.reshape(
                self.linear_o(
                    F.reshape(
                        F.tanh(
                            state_alignment_factor_broadcast +
                            precomputed_alignment_factor
                        ),
                        (
                            minibatch_size * max_sentence_size,
                            self.attention_layer_size
                        )
                    )
                ),
                (minibatch_size, max_sentence_size)
            )
            attention = F.softmax(scores)
            assert attention.shape == (minibatch_size, max_sentence_size)

            context = F.reshape(
                F.batch_matmul(attention, encoded, transa=True),
                (minibatch_size, encoder_output_size)
            )
            return context

        return compute_context
