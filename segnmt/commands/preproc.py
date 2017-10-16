from segnmt.preproc.preproc import preproc


def define_parser(parser):
    """Define command specific options."""
    parser.add_argument('source', type=str,
                        help='path of source document')
    parser.add_argument('target', type=str,
                        help='path of target document')
    parser.add_argument('output', type=str,
                        help='path of output directory')
    parser.add_argument('--max-source-len', type=int,
                        help='')
    parser.add_argument('--max-target-len', type=int,
                        help='')
    parser.add_argument('--source-dev', type=str,
                        help='')
    parser.add_argument('--target-dev', type=str,
                        help='')


def run(args):
    """Run the command."""
    preproc(args)
