from typing import List

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Parameter
from chainer import Variable

from segnmt.misc.constants import EOS
from segnmt.models.attention import AttentionModule


class Decoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 attention_hidden_layer_size: int,
                 encoder_output_size: int,
                 maxout_layer_size: int,
                 maxout_pool_size: int = 2,
                 ignore_label: int = -1):
        super(Decoder, self).__init__()
        with self.init_scope():
            self.embed_id = L.EmbedID(vocabulary_size,
                                      word_embeddings_size,
                                      ignore_label=ignore_label)
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
            self.bos_state = Parameter(
                initializer=self.xp.random.randn((1, hidden_layer_size))
            )
        self.vocabulary_size = vocabulary_size
        self.word_embeddings_size = word_embeddings_size
        self.hidden_layer_size = hidden_layer_size
        self.encoder_output_size = encoder_output_size

    def __call__(self,
                 encoded_source: Variable,
                 source_masks: List[Variable],
                 targets: List[Variable]) -> Variable:
        minibatch_size, max_sentence_size, encoder_output_size = \
            encoded_source.shape[0]
        assert encoder_output_size == self.encoder_output_size
        assert len(source_masks) == max_sentence_size
        assert source_masks[0].shape == (minibatch_size,)
        assert targets[0].shape == (minibatch_size,)

        compute_context = self.attention(encoded_source, source_masks)
        state = Variable(
            F.broadcast_to(
                self.bos_state, (minibatch_size, self.hidden_layer_size)
            )
        )
        total_loss = Variable(self.xp.zeros(minibatch_size))
        total_predictions = 0

        for target in targets:
            previous_output = self.embed_id(target)
            context = compute_context(state)
            assert context.shape == (minibatch_size, self.encoder_output_size)
            concatenated = F.concat((previous_output, context))
            state = self.rnn(concatenated)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))

            loss = F.softmax_cross_entropy(logit, target)
            total_loss += loss * minibatch_size
            total_predictions += minibatch_size

        return total_loss / total_predictions

    def translate(self,
                  encoded_source: Variable,
                  source_masks: List[Variable],
                  max_length: int = 100) -> List[Variable]:
        sentence_count = len(source_masks)
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            mask = F.vstack(source_masks)
            assert encoded_source.shape == mask.shape

            compute_context = self.attention(encoded_source, source_masks)
            state = Variable(
                F.broadcast_to(
                    self.bos_state, (sentence_count, self.hidden_layer_size)
                )
            )
            previous_id = self.xp.full((sentence_count,), EOS, 'i')
            result = []

            for i in range(max_length):
                previous_embedding = self.embed_id(previous_id)
                context = compute_context(state)
                assert context.shape == \
                       (sentence_count, self.encoder_output_size)
                concatenated = F.concat((previous_embedding, context))
                state = self.rnn(concatenated)
                all_concatenated = F.concat((concatenated, state))
                logit = self.linear(self.maxout(all_concatenated))

                previous_id = F.softmax(logit)
                result.append(previous_id)

            # Remove EOS tags
            outputs = F.separate(F.vstack(result), axis=0)
            output_sentences = []
            for output in outputs:
                indexes = self.xp.argwhere(output == EOS)
                if len(indexes) > 0:
                    output = output[indexes[0, 0]]
                output_sentences.append(output)

            return output_sentences
