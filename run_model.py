__author__ = 'Richard Diehl Martinez'
''' Entry point for launching problyglot '''

import click

from src.utils import setup
from src.problyglot import Problyglot

# Entry point
@click.command()
@click.argument('config_fp')
def main(config_fp):
    config = setup(config_fp)
    problyglot = Problyglot(config)
    problyglot()

if __name__ == '__main__':
    main()