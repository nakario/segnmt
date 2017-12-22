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


class SimilarityScoreFunction(chainer.Chain):
    def __init__(self, in_size: int):
        super(SimilarityScoreFunction, self).__init__()
        with self.init_scope():
            self.M = Parameter(chainer.initializers.Identity())
            if in_size is not None:
                self.M.initialize((in_size, in_size))
            self.l = Parameter(chainer.initializers.Zero(), ())

    def __call__(
            self,
            context: Variable,
            associated_contexts: Variable,
            beta: Variable
    ) -> Variable:
        minibatch_size, encoder_output_size = context.shape
        _, context_memory_size, _ = associated_contexts.shape
        assert associated_contexts.shape == (
            minibatch_size, context_memory_size, encoder_output_size
        )
        assert beta.shape == (minibatch_size, context_memory_size)

        if self.M.array is None:
            self.M.initialize((encoder_output_size, encoder_output_size))
        if self.l.array is None:
            self.l.initialize(())

        return F.squeeze(
            F.matmul(
                associated_contexts,
                F.expand_dims(F.linear(context, self.M), axis=2)
            ),
            axis=2
        ) - F.scale(beta, self.l, axis=0)


class GateFunction(chainer.Chain):
    def __init__(
            self,
            in_size: int,
            gate_hidden_layer_size: int
    ):
        super(GateFunction, self).__init__()
        with self.init_scope():
            self.linear_in = L.Linear(in_size, gate_hidden_layer_size)
            self.linear_out = L.Linear(gate_hidden_layer_size, 1)

    def __call__(
            self,
            context: Variable,
            state: Variable,
            averaged_state: Variable
    ) -> Variable:
        assert context.ndim == state.ndim == averaged_state.ndim == 2
        assert context.shape[0] == state.shape[0] == averaged_state.shape[0]
        return F.sigmoid(F.squeeze(
            self.linear_out(F.tanh(self.linear_in(
                F.concat((context, state, averaged_state), axis=1)
            ))),
            axis=1
        ))


class Decoder(chainer.Chain):
    def __init__(self,
                 vocabulary_size: int,
                 word_embeddings_size: int,
                 hidden_layer_size: int,
                 attention_hidden_layer_size: int,
                 gate_hidden_layer_size: int,
                 encoder_output_size: int,
                 maxout_layer_size: int,
                 maxout_pool_size: int = 2,
                 ignore_label: int = -1,
                 mode: str = 'deep'):
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
                                             hidden_layer_size,
                                             word_embeddings_size)
            self.bos_state = Parameter(
                initializer=self.xp.random.randn(
                    1,
                    hidden_layer_size
                ).astype('f')
            )
            self.E = SimilarityScoreFunction(encoder_output_size)
            self.compute_gate = GateFunction(
                encoder_output_size + hidden_layer_size + hidden_layer_size,
                gate_hidden_layer_size
            )
        self.vocabulary_size = vocabulary_size
        self.word_embeddings_size = word_embeddings_size
        self.hidden_layer_size = hidden_layer_size
        self.encoder_output_size = encoder_output_size
        assert mode in ['deep', 'shallow']
        self.mode = mode
        self.gate_sum = None
        self.averaged_gate = None
        self.averaged_beta = None
        self.max_score = None

    def __call__(
            self,
            encoded: Variable,
            target: ndarray,
            context_memory: Optional[
                Tuple[ndarray, ndarray, ndarray]
            ] = None
    ) -> Variable:
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size
        assert target.shape[0] == minibatch_size

        if self.bos_state.array is None:
            self.bos_state.initialize((1, self.hidden_layer_size))

        self.gate_sum = self.xp.zeros((minibatch_size,), 'f')
        self.max_score = self.xp.zeros((), 'f')
        self.attention.precompute(encoded)
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

        beta = None
        if context_memory is not None:
            context_memory_size = context_memory[0].shape[1]
            beta = Variable(
                self.xp.zeros((minibatch_size, context_memory_size), 'f')
            )

        for target_id in self.xp.hsplit(target, target.shape[1]):
            target_id = target_id.reshape((minibatch_size,))
            context = self.attention(state, previous_embedding)
            assert context.shape == (minibatch_size, self.encoder_output_size)
            concatenated = F.concat((previous_embedding, context))
            cell, state = self.rnn(cell, state, concatenated)

            if context_memory is not None and self.mode == 'deep':
                state, beta = \
                    self.fusion_state(context_memory, context, state, beta)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))
            if context_memory is not None and self.mode == 'shallow':
                logit, beta = self.fusion_logit(
                    context_memory, context, state, logit, beta
                )

            current_sentence_count = self.xp.sum(target_id != PAD)

            loss = F.softmax_cross_entropy(logit, target_id, ignore_label=PAD)
            total_loss += loss * current_sentence_count
            total_predictions += current_sentence_count

            previous_embedding = self.embed_id(target_id)

        if context_memory is not None:
            self.averaged_gate = \
                F.average(self.gate_sum / float(target.shape[1]))
            self.averaged_beta = F.average(beta)

        return total_loss / total_predictions

    def fusion_state(
            self,
            context_memory: Tuple[ndarray, ndarray, ndarray],
            context: Variable,
            state: Variable,
            beta: Variable
    ) -> Tuple[Variable, Variable]:
        minibatch_size = state.shape[0]
        associated_contexts = context_memory[0]
        context_memory_size = associated_contexts.shape[1]
        assert associated_contexts.shape == (
            minibatch_size,
            context_memory_size,
            self.encoder_output_size
        )
        associated_states = context_memory[1]
        assert associated_states.shape == (
            minibatch_size,
            context_memory_size,
            self.hidden_layer_size
        )

        assert beta.shape == (minibatch_size, context_memory_size)

        matching_score = F.softmax(
            self.E(context, Variable(associated_contexts), beta), axis=1
        )
        assert matching_score.shape == \
            (minibatch_size, context_memory_size)
        self.max_score = self.xp.maximum(
            self.max_score,
            self.xp.max(matching_score.array)
        )

        averaged_state = F.sum(
            F.scale(
                associated_states,
                matching_score,
                axis=0
            ),
            axis=1
        )
        assert averaged_state.shape == state.shape

        gate = self.compute_gate(context, state, averaged_state)
        assert gate.shape == (minibatch_size,)
        if self.gate_sum is not None:
            self.gate_sum += gate.array
        else:
            self.gate_sum = gate.array

        fusion = F.scale(averaged_state, gate, axis=0) \
            + F.scale(state, (1. - gate), axis=0)

        new_beta = beta + F.scale(matching_score, gate, axis=0)

        return fusion, new_beta

    def fusion_logit(
            self,
            context_memory: Tuple[ndarray, ndarray, ndarray],
            context: Variable,
            state: Variable,
            logit: Variable,
            beta: Variable
    ) -> Tuple[Variable, Variable]:
        minibatch_size = state.shape[0]
        associated_contexts = context_memory[0]
        context_memory_size = associated_contexts.shape[1]
        assert associated_contexts.shape == (
            minibatch_size,
            context_memory_size,
            self.encoder_output_size
        )
        associated_states = context_memory[1]
        assert associated_states.shape == (
            minibatch_size,
            context_memory_size,
            self.hidden_layer_size
        )
        associated_indices = context_memory[2]
        assert associated_indices.shape == (
            minibatch_size,
            context_memory_size
        )

        assert beta.shape == (minibatch_size, context_memory_size)

        matching_score = F.softmax(
            self.E(context, Variable(associated_contexts), beta), axis=1
        )
        assert matching_score.shape == \
               (minibatch_size, context_memory_size)
        self.max_score = self.xp.maximum(
            self.max_score,
            self.xp.max(matching_score.array)
        )

        averaged_state = F.sum(
            F.scale(
                associated_states,
                matching_score,
                axis=0
            ),
            axis=1
        )
        assert averaged_state.shape == state.shape

        gate = self.compute_gate(context, state, averaged_state)
        assert gate.shape == (minibatch_size,)
        if self.gate_sum is not None:
            self.gate_sum += gate.array
        else:
            self.gate_sum = gate.array

        flatten_indices = (
                associated_indices +
                self.xp.arange(
                    minibatch_size
                ).reshape((-1, 1)) * self.vocabulary_size
        ).flatten()
        assert flatten_indices.shape == (minibatch_size * context_memory_size,)
        indices_mask = (associated_indices != PAD).flatten()
        assert indices_mask.shape == flatten_indices.shape

        masked_score = F.where(
            indices_mask,
            F.flatten(F.scale(matching_score, gate, axis=0)),
            self.xp.zeros_like(flatten_indices, 'f')
        )

        fusion = F.log(F.reshape(
            F.scatter_add(
                F.flatten(
                    F.scale(F.softmax(logit, axis=1), (1. - gate), axis=0)
                ),
                flatten_indices,
                masked_score
            ),
            (minibatch_size, self.vocabulary_size)
        ))

        new_beta = beta + F.scale(matching_score, gate, axis=0)

        return fusion, new_beta

    def generate_keys(
            self,
            encoded: Variable,
            target: ndarray
    ) -> List[Tuple[ndarray, ndarray, ndarray]]:
        minibatch_size, max_sentence_size, encoder_output_size = encoded.shape
        assert encoder_output_size == self.encoder_output_size
        assert target.shape[0] == minibatch_size

        if self.bos_state.array is None:
            self.bos_state.initialize((1, self.hidden_layer_size))

        self.attention.precompute(encoded)
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
            context = self.attention(state, previous_embedding)
            assert context.shape == (minibatch_size, self.encoder_output_size)
            concatenated = F.concat((previous_embedding, context))
            cell, state = self.rnn(cell, state, concatenated)
            previous_embedding = self.embed_id(target_id)
            keys.append((context.data, state.data, target_id))

        return keys

    def translate(
            self,
            encoded: Variable,
            max_length: int = 100,
            context_memory: Optional[
                Tuple[ndarray, ndarray, ndarray]
            ] = None
    ) -> List[ndarray]:
        sentence_count = encoded.shape[0]

        if self.bos_state.array is None:
            self.bos_state.initialize((1, self.hidden_layer_size))

        self.gate_sum = self.xp.zeros((sentence_count,), 'f')
        self.max_score = self.xp.zeros((), 'f')
        self.attention.precompute(encoded)
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

        beta = None
        if context_memory is not None:
            context_memory_size = context_memory[0].shape[1]
            beta = Variable(
                self.xp.zeros((sentence_count, context_memory_size), 'f')
            )

        for _ in range(max_length):
            context = self.attention(state, previous_embedding)
            assert context.shape == \
                (sentence_count, self.encoder_output_size)
            concatenated = F.concat((previous_embedding, context))

            cell, state = self.rnn(cell, state, concatenated)

            if context_memory is not None and self.mode == 'deep':
                state, beta = \
                    self.fusion_state(context_memory, context, state, beta)
            all_concatenated = F.concat((concatenated, state))
            logit = self.linear(self.maxout(all_concatenated))
            if context_memory is not None and self.mode == 'shallow':
                logit, beta = self.fusion_logit(
                    context_memory, context, state, logit, beta
                )

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
