from segnmt.preproc.preproc import preproc
from segnmt.search_engine.similarity import functions


def define_parser(parser):
    """Define command specific options."""
    parser.add_argument('source', type=str,
                        help='path of source document')
    parser.add_argument('target', type=str,
                        help='path of target document')
    parser.add_argument('output', type=str,
                        help='path of output directory')
    parser.add_argument('--min-source-len', type=int, default=1,
                        help='')
    parser.add_argument('--max-source-len', type=int, default=70,
                        help='')
    parser.add_argument('--min-target-len', type=int, default=1,
                        help='')
    parser.add_argument('--max-target-len', type=int, default=70,
                        help='')
    parser.add_argument('--source-dev', type=str,
                        help='')
    parser.add_argument('--target-dev', type=str,
                        help='')
    parser.add_argument('--source-test', type=str,
                        help='')
    parser.add_argument('--target-test', type=str,
                        help='')
    parser.add_argument('--skip-create-index', action='store_true',
                        help='')
    parser.add_argument('--skip-sleep', action='store_true',
                        help='')
    parser.add_argument('--skip-make-sim', action='store_true',
                        help='')
    parser.add_argument('--skip-create-bpe', action='store_true',
                        help='')
    parser.add_argument('--skip-bpe-encode', action='store_true',
                        help='')
    parser.add_argument('--skip-make-voc', action='store_true',
                        help='')
    parser.add_argument('--limit', type=int, default=-1,
                        help='')
    parser.add_argument('--similarity-function', type=str,
                        choices=functions.keys(),
                        default=list(functions.keys())[0],
                        help='')
    parser.add_argument('--sleep-time', type=int, default=60,
                        help='how many seconds it sleeps')


def run(args):
    """Run the command."""
    preproc(args)
