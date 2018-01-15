import argparse
from logging import getLogger
from typing import Generator
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import chainer
from chainer import Variable
import chainer.functions as F
import matplotlib
from nltk.translate import bleu_score
import numpy as np
from progressbar import ProgressBar

from segnmt.misc.constants import EOS
from segnmt.misc.typing import ndarray
from segnmt.models.encdec import EncoderDecoder
from segnmt.train.train import decode_bpe
from segnmt.train.train import convert
from segnmt.train.train import convert_with_similar_sentences
from segnmt.train.train import load_vocab
from segnmt.train.train import load_data
from segnmt.train.train import load_validation_data


logger = getLogger(__name__)
matplotlib.use('Agg')


class ConstArguments(NamedTuple):
    # Encoder-Decoder arguments
    source_vocabulary_size: int
    source_word_embeddings_size: int
    encoder_hidden_layer_size: int
    encoder_num_steps: int
    encoder_dropout: float
    target_vocabulary_size: int
    target_word_embeddings_size: int
    decoder_hidden_layer_size: int
    attention_hidden_layer_size: int
    gate_hidden_layer_size: int
    maxout_layer_size: int
    fusion_mode: str

    gpu: int
    minibatch_size: int
    source_vocab: str
    target_vocab: str
    training_source: str
    training_target: str
    validation_source: str
    validation_target: str
    similar_sentence_indices: Optional[str]
    similar_sentence_indices_validation: Optional[str]
    translation_output_file: str
    resume_files: str
    max_translation_length: int
    beam_width: int


class ModelState(NamedTuple):
    cell: Variable
    hidden: Variable
    beta: Optional[Variable]


class TranslationStates(NamedTuple):
    translations: List
    scores: ndarray
    states: List[ModelState]
    words: ndarray


def compute_next_states_and_scores(
        models: List[EncoderDecoder],
        states: List[ModelState],
        words: ndarray
) -> Tuple[ndarray, List[ModelState]]:
    xp = models[0].xp
    cells, hiddens, contexts, concatenateds = zip(*[model.dec.advance_one_step(state.cell, state.hidden, words) for model, state in zip(models, states)])
    logits, hiddens, betas = zip(*[model.dec.compute_logit(concatenated, hidden, context, state.beta) for (model, concatenated, context, state, hidden) in zip(models, concatenateds, contexts, states, hiddens)])

    combined_scores = xp.zeros_like(logits[0].array, 'f')

    for logit in logits:
        combined_scores += xp.log(F.softmax(logit).array)
    combined_scores /= float(len(models))

    new_states = []
    for cell, hidden, beta in zip(cells, hiddens, betas):
        new_states.append(ModelState(cell, hidden, beta))

    return combined_scores, new_states


def iterate_best_score(scores: ndarray, beam_width: int) -> Generator[Tuple[int, int, float]]:
    case_, voc_size = scores.shape

    costs_flattened: np.ndarray = chainer.cuda.to_cpu(-scores).ravel()
    best_index = np.argpartition(costs_flattened, beam_width)[:beam_width]

    which_case = np.floor_divide(best_index, voc_size)
    index_in_case = best_index % voc_size

    for i, idx in enumerate(best_index):
        case = which_case[i]
        idx_in_case = index_in_case[i]
        yield int(case), int(idx_in_case), float(costs_flattened[idx])


def update_next_lists(
        case: int,
        idx_in_case: int,
        cost: float,
        states: List[ModelState],
        translations: List[List[int]],
        finished_translations: List,
        next_states_list: List[List[ModelState]],
        next_words_list: List[int],
        next_score_list: List[float],
        next_translations_list: List[List[int]]
):
    if idx_in_case == EOS:
        finished_translations.append((translations[case] + [idx_in_case], -cost))
    else:
        next_states_list.append([
            ModelState(
                Variable(state.cell.array[case].reshape(1, -1)),
                Variable(state.hidden.array[case].reshape(1, -1)),
                Variable(state.beta.array[case].reshape(1, -1))
            ) for state in states
        ])
        next_words_list.append(idx_in_case)
        next_score_list.append(-cost)
        next_translations_list.append(translations[case] + [idx_in_case])


def compute_next_lists(
        states: List[ModelState],
        translations: List[List[int]],
        scores: ndarray,
        finished_translations,
        beam_width: int
) -> Tuple[List[List[ModelState]], List[int], List[float], List[List[int]]]:
    next_states_list = []
    next_words_list = []
    next_score_list = []
    next_translations_list = []

    score_iterator = iterate_best_score(scores, beam_width)

    for case, idx_in_case, cost in score_iterator:
        update_next_lists(case, idx_in_case, cost, states, translations, finished_translations, next_states_list, next_words_list, next_score_list, next_translations_list)

    return next_states_list, next_words_list, next_score_list, next_translations_list


def advance_one_step(
        models: List[EncoderDecoder],
        translation_states: TranslationStates,
        finished_translations: List,
        beam_width: int
) -> TranslationStates:
    xp = models[0].xp
    translations = translation_states.translations
    scores = translation_states.scores
    states = translation_states.states
    words = translation_states.words
    combined_scores, new_states = compute_next_states_and_scores(models, states, words)

    count, voc_size = combined_scores.shape
    assert count <= beam_width

    word_scores = scores[:, None] + combined_scores

    next_states_list, next_words_list, next_score_list, next_translations_list = compute_next_lists(new_states, translations, word_scores, finished_translations, beam_width)

    concatenated_next_states_list = []
    for ss in zip(*next_states_list): # loops for len(models) times
        # concat beams
        cell = F.concat([s.cell for s in ss], axis=0)
        hidden = F.concat([s.hidden for s in ss], axis=0)
        beta = F.concat([s.beta for s in ss], axis=0)
        concatenated_next_states_list.append(
            ModelState(cell, hidden, beta)
        )

    return TranslationStates(
        next_translations_list,
        xp.array(next_score_list, 'f'),
        concatenated_next_states_list,
        xp.array(next_words_list, 'i')
    )


def translate_ensemble(
        models: List[EncoderDecoder],
        source: ndarray,
        similars: Optional[List[Tuple[ndarray, ndarray]]],
        translation_limit: int,
        beam_width: int
) -> List[ndarray]:
    xp = chainer.cuda.get_array_module(source)
    assert source.shape[0] == 1
    encodeds = [model.enc(source) for model in models]
    context_memories = None
    if similars is not None:
        context_memories = [
            model.generate_context_memory(similars) for model in models
        ]

    states: List[ModelState] = []
    finished_translations: List[Tuple[List[int], float]] = []
    previous_words = None # shared in all models
    for model, encoded, context_memory in zip(
            models, encodeds, context_memories
    ):
        model.dec.setup(encoded, context_memory)
        cell, hidden, previous_words, beta = model.dec.get_initial_states(1)
        states.append(ModelState(cell, hidden, beta))

    translation_states = TranslationStates([[]], xp.array([0], 'f'), states, previous_words)
    for i in range(translation_limit):
        translation_states = advance_one_step(models, translation_states, finished_translations, beam_width)

    if len(finished_translations) == 0:
        finished_translations.append(([], 0))

    finished_translations.sort(key=lambda x: x[1] / (len(x[0]) + 2), reverse=True)

    return [xp.array(finished_translations[0][0])]


def evaluate(args: argparse.Namespace):
    cargs = ConstArguments(**vars(args))
    logger.info(f'cargs: {cargs}')

    models = []
    for model_file in cargs.resume_files.split(sep=','):
        model = EncoderDecoder(cargs.source_vocabulary_size,
                               cargs.source_word_embeddings_size,
                               cargs.encoder_hidden_layer_size,
                               cargs.encoder_num_steps,
                               cargs.encoder_dropout,
                               cargs.target_vocabulary_size,
                               cargs.target_word_embeddings_size,
                               cargs.decoder_hidden_layer_size,
                               cargs.attention_hidden_layer_size,
                               cargs.gate_hidden_layer_size,
                               cargs.maxout_layer_size,
                               cargs.fusion_mode)
        if cargs.gpu >= 0:
            chainer.cuda.get_device_from_id(cargs.gpu).use()
            model.to_gpu(cargs.gpu)

        chainer.serializers.load_npz(model_file, model)

        models.append(model)

    source_vocab = load_vocab(cargs.source_vocab, cargs.source_vocabulary_size)
    target_vocab = load_vocab(cargs.target_vocab, cargs.target_vocabulary_size)

    converter = convert
    if cargs.similar_sentence_indices is not None:
        converter = convert_with_similar_sentences

    if cargs.similar_sentence_indices_validation is not None:
        validation_data = load_validation_data(
            cargs.training_source,
            cargs.training_target,
            cargs.validation_source,
            cargs.validation_target,
            source_vocab,
            target_vocab,
            cargs.similar_sentence_indices_validation,
            1000
        )
    else:
        validation_data = load_data(
            cargs.validation_source,
            cargs.validation_target,
            source_vocab,
            target_vocab
        )

    v_iter = chainer.iterators.SerialIterator(
        validation_data,
        1,
        repeat=False,
        shuffle=False
    )

    target_sentences: List[List[List[str]]]
    with open(cargs.validation_target) as f:
        target_sentences = \
            list(map(lambda x: [x.strip().split()], f.readlines()))

    target_word = {index: word for word, index in target_vocab.items()}

    list_of_references: List[List[List[str]]] = []
    hypotheses: List[List[str]] = []
    v_iter.reset()
    print("start translation")
    with chainer.no_backprop_mode(), chainer.using_config('train', False):
        i = 0
        bar = ProgressBar()
        for minibatch in bar(v_iter, max_value=len(target_sentences)):
            list_of_references.extend(target_sentences[i:i+1])
            i += 1
            converted = converter(minibatch, cargs.gpu)
            source = converted[0]
            similars = None
            if len(converted) == 3:
                similars = converted[2]
            results = translate_ensemble(
                models,
                source,
                similars,
                translation_limit=cargs.max_translation_length,
                beam_width=cargs.beam_width
            )
            hypotheses.extend([
                decode_bpe([
                    target_word.get(id_, '<UNK>')
                    for id_ in sentence.tolist()[:-1]
                ]) for sentence in results
            ])
    print("start write file")
    assert len(list_of_references) == len(hypotheses)
    with open(cargs.translation_output_file, 'w') as output:
        for i in range(len(list_of_references)):
            output.write(f"src: {' '.join(list_of_references[i][0])}\n")
            output.write(f"out: {' '.join(hypotheses[i])}\n\n")
    print("start calc bleu")
    bleu = bleu_score.corpus_bleu(
        list_of_references,
        hypotheses
    )
    print(f"BLEU: {bleu}")
