"""
module main
"""

import argparse
from logging import basicConfig
from logging import INFO
from typing import List
from typing import Optional

import segnmt.commands.preproc as preproc
import segnmt.commands.train as train

basicConfig(level=INFO)


def main(arguments: Optional[List[str]] = None):
    """Define command line parser and run the specified command."""
    parser = argparse.ArgumentParser(prog='segnmt')
    parser.set_defaults(run=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()

    preproc_parser = subparsers.add_parser('preproc')
    preproc.define_parser(preproc_parser)
    preproc_parser.set_defaults(run=preproc.run)

    train_parser = subparsers.add_parser('train')
    train.define_parser(train_parser)
    train_parser.set_defaults(run=train.run)

    args = parser.parse_args(args=arguments)
    run = args.run
    delattr(args, 'run')
    run(args)


if __name__ == '__main__':
    main()
