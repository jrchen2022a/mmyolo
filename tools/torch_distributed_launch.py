import logging
import warnings

from torch.distributed.run import get_args_parser, run


logger = logging.getLogger(__name__)


def parse_args(args):
    parser = get_args_parser()
    parser.add_argument(
        "--use_env",
        default=False,
        action="store_true",
        help="Use environment variable to pass "
        "'local rank'. For legacy reasons, the default value is False. "
        "If set to True, the script will not pass "
        "--local_rank as argument, and will instead set LOCAL_RANK.",
    )
    return parser.parse_args(args)


def launch(args):
    if args.no_python and not args.use_env:
        raise ValueError(
            "When using the '--no_python' flag,"
            " you must also set the '--use_env' flag."
        )
    run(args)


def main(args=None):
    warnings.warn(
        "The module torch.distributed.launch is deprecated\n"
        "and will be removed in future. Use torchrun.\n"
        "Note that --use_env is set by default in torchrun.\n"
        "If your script expects `--local_rank` argument to be set, please\n"
        "change it to read from `os.environ['LOCAL_RANK']` instead. See \n"
        "https://pytorch.org/docs/stable/distributed.html#launch-utility for \n"
        "further instructions\n",
        FutureWarning,
    )
    args = parse_args(args)
    launch(args)


if __name__ == "__main__":
    main()