"""
module main
"""

import argparse
from importlib import import_module
from pkgutil import iter_modules
from typing import List
from typing import Optional

import segnmt.commands


def main(arguments: Optional[List[str]] = None):
    """Define command line parser and run the specified command."""
    parser = argparse.ArgumentParser(prog='segnmt')
    subparsers = parser.add_subparsers(dest='command')
    commands = segnmt.commands
    for _, command, is_pkg in iter_modules(commands.__path__):
        assert not is_pkg
        command_parser = subparsers.add_parser(command)
        command_module = import_module(f"{commands.__name__}.{command}")
        command_module.define_parser(command_parser)
        command_parser.set_defaults(run=command_module.run)
    args = parser.parse_args(args=arguments)
    args.run(args)


if __name__ == '__main__':
    main()
