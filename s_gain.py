# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main interface to interact with the S-GAIN testing framework."""

import argparse

import config

import utils.subroutines as subroutines


# -- Main -------------------------------------------------------------------------------------------------------------

def main(args):
    """The main interface to interact with S-GAIN."""

    # Select subroutine
    subroutine = args.subroutine
    if subroutine == 'settings':
        subroutines.settings(settings, args.operation, args.filename, args.information)
    elif subroutine == 'run':
        subroutines.run_experiments(config)
    elif subroutine == 'analyze':
        subroutines.analyze(config, args.input, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    # Parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='subroutine',
        title='subroutine',
        help='change settings, run or analyze experiments'
    )

    # Settings
    settings = subparsers.add_parser(
        'settings',
        help='show, load, store or delete settings'
    )
    settings.add_argument(
        'operation',
        choices=['show', 'list', 'ls', 'load', 'l', 'store', 'save', 's', 'delete', 'del', 'remove', 'rm'],
        help='show settings, load settings (default, IDEAL2025, showcase, ...), store current settings or delete settings',
        type=str
    )
    settings.add_argument(
        'filename',
        help='the name of the settings file to show, load, store or delete (shows the current settings if left blank)',
        nargs='?',
        type=str
    )
    settings.add_argument(
        '--information', '-info',
        help='show additional information about the settings',
        action='store_true'
    )


    # Run experiments
    run = subparsers.add_parser(
        'run',
        help='run the experiments specified in config.py'
    )
    # Todo overwrite config

    # Analysis
    analysis = subparsers.add_parser(
        'analyze',
        help='analyze the completed experiments'
    )
    analysis.add_argument(
        'input',
        help='the folder where the completed experiments are located (use default: output, if not specified)',
        nargs='?',
        type=str
    )
    analysis.add_argument(
        '-in', '--input',
        help='the folder where the completed experiments are located (use default: output, if not specified)',
        type=str
    )
    analysis.add_argument(
        'output',
        help='the folder to save the analysis to (use default: analysis, if not specified)',
        nargs='?',
        type=str
    )
    analysis.add_argument(
        '-out', '--output',
        help='the folder to save the analysis to (use default: analysis, if not specified)',
        type=str
    )
    # Todo overwrite config

    # Call main
    args = parser.parse_args()
    main(args)
