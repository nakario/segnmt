import argparse
from typing import Callable
from typing import NamedTuple
from typing import Optional

from segnmt.train.train import train


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
    maxout_layer_size: int

    gpu: int
    minibatch_size: int
    epoch: int
    source_vocab: str
    target_vocab: str
    training_source: str
    training_target: str
    validation_source: Optional[str]
    validation_target: Optional[str]
    min_source_len: int
    max_source_len: int
    min_target_len: int
    max_target_len: int

    run: Callable[[argparse.Namespace], None]


def define_parser(parser: argparse.ArgumentParser):
    """Define command specific options."""
    parser.add_argument('--source-vocabulary-size', type=int, default=40000,
                        help='The number of words of source language')
    parser.add_argument('--source-word-embeddings-size', type=int, default=640,
                        help='')
    parser.add_argument('--encoder-hidden-layer-size', type=int, default=1024,
                        help='')
    parser.add_argument('--encoder-num-steps', type=int, default=1,
                        help='')
    parser.add_argument('--encoder-dropout', type=float, default=0.1,
                        help='')
    parser.add_argument('--target-vocabulary-size', type=int, default=40000,
                        help='')
    parser.add_argument('--target-word-embeddings-size', type=int, default=640,
                        help='')
    parser.add_argument('--decoder-hidden-layer-size', type=int, default=1024,
                        help='')
    parser.add_argument('--attention-hidden-layer-size', type=int,
                        default=1024, help='')
    parser.add_argument('--maxout-layer-size', type=int, default=512,
                        help='')

    parser.add_argument('--gpu', type=int, default=-1,
                        help='GPU ID (-1 means CPU)')
    parser.add_argument('--minibatch-size', type=int, default=64,
                        help='')
    parser.add_argument('--epoch', type=int, default=20,
                        help='')
    parser.add_argument('--source-vocab', type=str, required=True,
                        help='')
    parser.add_argument('--target-vocab', type=str, required=True,
                        help='')
    parser.add_argument('--training-source', type=str, required=True,
                        help='')
    parser.add_argument('--training-target', type=str, required=True,
                        help='')
    parser.add_argument('--validation-source', default=None,
                        help='')
    parser.add_argument('--validation-target', default=None,
                        help='')
    parser.add_argument('--min-source-len', type=int, default=1,
                        help='')
    parser.add_argument('--max-source-len', type=int, default=50,
                        help='')
    parser.add_argument('--min-target-len', type=int, default=1,
                        help='')
    parser.add_argument('--max-target-len', type=int, default=50,
                        help='')


def run(args: argparse.Namespace):
    """Run the command."""
    cargs = ConstArguments(**vars(args))
    train(cargs)
