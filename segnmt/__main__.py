"""
module main
"""

import argparse
from importlib import import_module
from logging import basicConfig
from logging import DEBUG
from logging import INFO
from pkgutil import iter_modules
from typing import List
from typing import Optional

import segnmt.commands


basicConfig(level=DEBUG)


def main(arguments: Optional[List[str]] = None):
    """Define command line parser and run the specified command."""
    parser = argparse.ArgumentParser(prog='segnmt')
    parser.set_defaults(run=lambda _: parser.print_help())
    subparsers = parser.add_subparsers()
    commands = segnmt.commands
    for _, command, is_pkg in iter_modules(commands.__path__):
        if is_pkg:
            continue
        command_parser = subparsers.add_parser(command)
        command_module = import_module(f"{commands.__name__}.{command}")
        command_module.define_parser(command_parser)
        command_parser.set_defaults(run=command_module.run)
    args = parser.parse_args(args=arguments)
    args.run(args)


if __name__ == '__main__':
    main()
