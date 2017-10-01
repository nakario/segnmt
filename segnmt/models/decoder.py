from logging import getLogger
from typing import List
from typing import Optional
from typing import Tuple

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Parameter
from chainer import Variable
import numpy as np

from segnmt.misc.constants import EOS
from segnmt.misc.constants import PAD
from segnmt.misc.typing import ndarray
from segnmt.models.attention import AttentionModule


logger = getLogger(__name__)


class SimilarityScoreFunction(chainer.Chain):
    def __init__(self, in_size: int):
        super(SimilarityScoreFunction, self).__init__()
        with self.init_scope():
            self.M = Parameter(chainer.initializers.Identity())
            if in_size is not None:
                self.M.initialize((in_size, in_size))
            self.l = Parameter(chainer.initializers.Zero(), 1)

    def __call__(
            self,
            context: Variable,
            associated_contexts: Variable,
            beta: Variable) -> Variable:
        minibatch_size, encoder_output_size = context.shape
        _, _, context_memory_size = associated_contexts.shape
        assert context.shape == associated_contexts.shape[:2]
        assert associated_contexts.shape[::2] == beta.shape
        if self.M.data is None:
            self.M.initialize((encoder_output_size, encoder_output_size))
        return F.reshape(
            F.batch_matmul(
                F.reshape(
                    F.linear(context, self.M),
                    (minibatch_size, 1, encoder_output_size)
                ),
                associated_contexts
            ),
            (minibatch_size, context_memory_size)
        ) - F.scale(beta, self.l)


class GateFunction(chainer.Chain):
    def __init__(self, in_size: int):
        super(GateFunction, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(in_size, 1)

    def __call__(
            self,
            context: Variable,
            state: Variable,
            averaged_state: Variable
    ) -> Variable:
        return F.sigmoid(
            self.linear(
                F.concat((context, state, averaged_state), axis=1)
            )
        )


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
            self.rnn = L.StatelessLSTM(
                word_embeddings_size + encoder_output_size,
                hidden_layer_size
            )
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
                initializer=self.xp.random.randn(
                    1,
                    hidden_layer_size
                ).astype('f')
            )
            self.E = SimilarityScoreFunction(encoder_output_size)
            self.compute_gate = GateFunction(
                encoder_output_size + hidden_layer_size + hidden_layer_size
            )
        self.vocabulary_size = vocabulary_size
        self.word_embeddings_size = word_embeddings_size
        self.hidden_layer_size = hidden_layer_size
        self.encoder_output_size = encoder_output_size

    def __call__(
            self,
            encoded: Variable,
            target: ndarray,
            context_memory: Optional[
                Tuple[ndarray, ndarray, ndarray, ndarray]
            ] = None
    ) -> Variable:
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size
        assert target.shape[0] == minibatch_size

        compute_context = self.attention(encoded)
        cell = Variable(
            self.xp.zeros((minibatch_size, self.hidden_layer_size), 'f')
        )
        state = F.broadcast_to(
            self.bos_state, (minibatch_size, self.hidden_layer_size)
        )
        previous_embedding = self.embed_id(
            Variable(self.xp.full((minibatch_size,), EOS, 'i'))
        )
        total_loss = Variable(self.xp.array(0, 'f'))
        total_predictions = 0

        for target_id in self.xp.hsplit(target, target.shape[1]):
            target_id = target_id.reshape((minibatch_size,))
            context = compute_context(state)
            assert context.shape == (minibatch_size, self.encoder_output_size)
            concatenated = F.concat((previous_embedding, context))
            cell, state = self.rnn(cell, state, concatenated)

            if context_memory is not None:
                associated_contexts = context_memory[0]
                context_memory_size = associated_contexts.shape[2]
                assert associated_contexts.shape == (
                    minibatch_size,
                    self.encoder_output_size,
                    context_memory_size
                )
                associated_states = context_memory[1]
                assert associated_states.shape == (
                    minibatch_size,
                    self.hidden_layer_size,
                    context_memory_size
                )
                associated_logit = context_memory[2]
                assert associated_logit.shape == (
                    minibatch_size,
                    self.vocabulary_size,
                    context_memory_size
                )
                beta = context_memory[3]
                assert beta.shape == (minibatch_size, context_memory_size)

                matching_score = F.softmax(
                    self.E(context, associated_contexts, beta), axis=1
                )
                assert matching_score.shape == \
                    (minibatch_size, context_memory_size)

                averaged_state = F.average(
                    F.broadcast_to(
                        F.reshape(
                            matching_score,
                            (
                                minibatch_size,
                                1,
                                context_memory_size
                            )
                        ),
                        (
                            minibatch_size,
                            self.hidden_layer_size,
                            context_memory_size
                        )
                    ) * associated_states,
                    axis=2
                )
                assert state.shape == averaged_state.shape
                gate = self.compute_gate(context, state, averaged_state)
                state = \
                    F.scale(averaged_state, gate, axis=0)\
                    + F.scale(state, (1. - gate), axis=0)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))

            current_sentence_count = self.xp.sum(target_id != PAD)

            loss = F.softmax_cross_entropy(logit, target_id, ignore_label=PAD)
            total_loss += loss * current_sentence_count
            total_predictions += current_sentence_count

            previous_embedding = self.embed_id(target_id)

        return total_loss / total_predictions

    def generate_keys(
            self,
            encoded: Variable,
            target: ndarray
    ) -> List[Tuple[ndarray, ndarray, ndarray]]:
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size
        assert target.shape[0] == minibatch_size

        compute_context = self.attention(encoded)
        cell = Variable(
            self.xp.zeros((minibatch_size, self.hidden_layer_size), 'f')
        )
        state = F.broadcast_to(
            self.bos_state, (minibatch_size, self.hidden_layer_size)
        )
        previous_embedding = self.embed_id(
            Variable(self.xp.full((minibatch_size,), EOS, 'i'))
        )
        keys = []

        for target_id in self.xp.hsplit(target, target.shape[1]):
            target_id = target_id.reshape((minibatch_size,))
            context = compute_context(state)
            assert context.shape == (minibatch_size, self.encoder_output_size)
            concatenated = F.concat((previous_embedding, context))
            cell, state = self.rnn(cell, state, concatenated)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))
            previous_embedding = self.embed_id(target_id)
            keys.append((context.data, state.data, logit.data))

        return keys

    def translate(self,
                  encoded: Variable, max_length: int = 10) -> List[ndarray]:
        sentence_count = encoded.shape[0]
        compute_context = self.attention(encoded)
        cell = Variable(
            self.xp.zeros((sentence_count, self.hidden_layer_size), 'f')
        )
        state = F.broadcast_to(
            self.bos_state, (sentence_count, self.hidden_layer_size)
        )
        previous_embedding = self.embed_id(
            Variable(self.xp.full((sentence_count,), EOS, 'i'))
        )
        result = []

        for _ in range(max_length):
            context = compute_context(state)
            assert context.shape == \
                (sentence_count, self.encoder_output_size)
            concatenated = F.concat((previous_embedding, context))

            cell, state = self.rnn(cell, state, concatenated)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))

            output_id = F.reshape(F.argmax(logit, axis=1), (sentence_count,))
            result.append(output_id)

            previous_embedding = self.embed_id(output_id)

        # Remove words after <EOS>
        outputs = F.separate(F.transpose(F.vstack(result)), axis=0)
        assert len(outputs) == sentence_count
        output_sentences = []
        for output in outputs:
            assert output.shape == (max_length,)
            indexes = np.argwhere(output.data == EOS)
            if len(indexes) > 0:
                output = output[:indexes[0, 0] + 1]
            output_sentences.append(output.data)

        return output_sentences
