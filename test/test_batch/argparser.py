import sys
import json
import argparse


def _initialize():
    parser = argparse.ArgumentParser(
        description="General config loader."
    )
    parser.add_argument(
        "-c",
        "--config_file",
        help="Address of the config file."
    )

    return parser


def _get_options(parser, args=sys.argv[1:]):
    options = parser.parse_args(args)
    return options


def parse_options(config_path=None):
    config_file = None
    if config_path is None:
        parser = _initialize()

        options = _get_options(parser)
        config_file = options.config_file

        if config_file is None:
            print('Please provide the config file.')
            parser.print_help()
            exit()
    else:
        config_file = config_path

    parameters = None
    with open(config_file) as json_file:
        parameters = json.load(json_file)

    return parameters
