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

"""The main interface to interact with S-GAIN.

(1) print_settings: print the specified settings to the console
(2) print_experiments: print the experiments the current settings would run to the console
"""

import argparse

from os.path import isfile

from analyze import analyze
from run_experiments import run_experiments


def print_settings(filename):
    """Print the specified settings to the console.

    :param filename: the name of the settings file
    """

    # Print specified settings
    if filename:
        file = f'settings/{filename}.py'
        if isfile(file):
            with open(file, 'r') as f:
                s = f.read().split('"""\n')[-1]
        else:
            s = f'{filename} settings not found!'

    # Print current settings
    else:
        with open('config.py', 'r') as f:
            s = f.read().split('"""\n')[-1]

    print(s)


def print_experiments():  # Todo show experiments
    """Print the experiments the current settings would run to the console."""

    print('print_experiments is not implemented yet')  # Todo show to run and completed


def main(parser, args):
    subroutine = args.subroutine

    # Settings subroutine
    if subroutine == 'settings':
        operation = args.operation
        filename = args.filename

        # Show settings
        if operation == 'show':
            print_settings(filename)

        # Load/store settings
        elif filename:
            file = f'settings/{filename}.py'

            # Load previous settings
            if operation == 'load':
                if isfile(file):
                    with open(file, 'r') as f:
                        cfg = f.read()
                    with open('config.py', 'w') as f:
                        f.write(cfg)
                else:
                    print(f'{filename} settings not found!')

            # Store current settings
            else:
                with open('config.py', 'r') as f:
                    cfg = f.read()
                with open(file, 'w') as f:
                    f.write(cfg)

        else:  # No filename specified
            parser.print_help()  # Todo better help

        # Todo set settings through terminal

    # Run experiments subroutine
    elif subroutine == 'run':
        run_experiments()

    # Analysis subroutine
    elif subroutine == 'analyze':
        analyze()

    # Help subroutine
    else:
        parser.print_help()



if __name__ == '__main__':
    # Parser and subparsers
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest='subroutine',
        title='subroutine'
    )

    # Settings
    settings = subparsers.add_parser('settings')
    settings.add_argument(
        'operation',
        choices=['show', 'load', 'store', ],
        help='show current settings, load previous settings (default, IDEAL2025, ...) or store current settings',
        type=str
    )
    settings.add_argument(
        'filename',
        help='the name of the settings file to show/load/save',
        nargs='?',
        type=str
    )

    # Run experiments
    run = subparsers.add_parser('run')

    # Analysis
    analysis = subparsers.add_parser('analyze')

    # Call main
    args = parser.parse_args()
    main(parser, args)
