import argparse
from typing import Callable
from typing import NamedTuple

from segnmt.train.train import train


class ConstArguments(NamedTuple):
    gpu: int
    run: Callable[[argparse.Namespace], None]


def define_parser(parser: argparse.ArgumentParser):
    """Define command specific options."""
    parser.add_argument('-g', '--gpu', type=int, default=-1,
                        help='GPU ID (-1 means CPU)')


def run(args: argparse.Namespace):
    """Run the command."""
    cargs = ConstArguments(**vars(args))
    train(cargs)
