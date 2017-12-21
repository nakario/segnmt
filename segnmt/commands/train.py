import argparse

from segnmt.train.train import train


def define_parser(parser: argparse.ArgumentParser):
    """Define command specific options.

    See `segnmt.train.train.ConstArguments`
    """
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
    parser.add_argument('--gate-hidden-layer-size', type=int,
                        default=512, help='')
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
    parser.add_argument('--similar-sentence-indices', default=None,
                        help='')
    parser.add_argument('--similar-sentence-indices-validation', default=None,
                        help='')
    parser.add_argument('--loss-plot-file', type=str, default='loss.png',
                        help='')
    parser.add_argument('--bleu-plot-file', type=str, default='bleu.png',
                        help='')
    parser.add_argument('--resume-file', default=None,
                        help='')
    parser.add_argument('--extension-trigger', type=int, default=100,
                        help='The number of iterations to trigger extensions')
    parser.add_argument('--similar-limit', type=int, default=2,
                        help='The number of similar sentences')
    parser.add_argument('--similar-limit-validation', type=int, default=4,
                        help='The number of similar sentences')
    parser.add_argument('--fusion-mode', choices=['deep', 'shallow'],
                        default='deep', help='')
    parser.add_argument('--max-translation-length', type=int, default=100,
                        help='')


def run(args: argparse.Namespace):
    """Run the command."""
    train(args)
