__author__ = 'Richard Diehl Martinez'
''' Entry point for launching problyglot '''

import click

from src.utils import setup
from src.problyglot import Problyglot

# Entry point
@click.command()
@click.argument('config_fp')
@click.option('--train-model', default=True)
def main(config_fp, train_model):
    config = setup(config_fp)

    problyglot = Problyglot(config)
    if train_model:
        problyglot.train()

if __name__ == '__main__':
    main()