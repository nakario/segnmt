import argparse

from segnmt.eval.eval import evaluate


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
    parser.add_argument('--source-vocab', type=str, required=True,
                        help='')
    parser.add_argument('--target-vocab', type=str, required=True,
                        help='')
    parser.add_argument('--training-source', type=str, required=True,
                        help='')
    parser.add_argument('--training-target', type=str, required=True,
                        help='')
    parser.add_argument('--validation-source', type=str, required=True,
                        help='')
    parser.add_argument('--validation-target', type=str, required=True,
                        help='')
    parser.add_argument('--similar-sentence-indices', default=None,
                        help='')
    parser.add_argument('--similar-sentence-indices-validation', default=None,
                        help='')
    parser.add_argument('--translation-output-file', type=str,
                        default='output.txt', help='')
    parser.add_argument('--resume-file', type=str, required=True,
                        help='best_bleu.npz')
    parser.add_argument('--fusion-mode', choices=['deep', 'shallow'],
                        default='deep', help='')
    parser.add_argument('--max-translation-length', type=int, default=100,
                        help='')


def run(args: argparse.Namespace):
    """Run the command."""
    evaluate(args)
