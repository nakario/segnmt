from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

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
            self.embed_id = L.EmbedID(vocabulary_size, word_embeddings_size)
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
        self.vocabulary_size = vocabulary_size
        self.word_embeddings_size = word_embeddings_size
        self.hidden_layer_size = hidden_layer_size
        self.encoder_output_size = encoder_output_size

    def __call__(self,
                 encoded_source: Variable,
                 source_masks: List[Variable],
                 targets: List[Variable]
    ) -> Variable:
        minibatch_size, max_sentence_size, encoder_output_size = \
            encoded_source.shape[0]
        assert encoder_output_size == self.encoder_output_size
        assert len(source_masks) == max_sentence_size
        assert source_masks[0].shape == (minibatch_size,)
        assert targets[0].shape == (minibatch_size,)

        compute_context = self.attention(encoded_source, source_masks)
        state = Variable(
            F.broadcast_to(
                self.xp.random.randn((1, self.hidden_layer_size)),
                (minibatch_size, self.hidden_layer_size)
            )
        )
        total_loss = Variable(self.xp.zeros(minibatch_size))
        total_predictions = 0

        for i in range(len(targets)):
            previous_output = self.embed_id(targets[i])
            context = compute_context(state)
            assert context.shape == (minibatch_size, self.encoder_output_size)
            concatenated = F.concat((previous_output, context))
            state = self.rnn(concatenated)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))

            loss = F.softmax_cross_entropy(logit, targets[i])
            total_loss += loss * minibatch_size
            total_predictions += minibatch_size

        return total_loss / total_predictions