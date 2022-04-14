__author__ = 'Richard Diehl Martinez'
''' Entry point for launching problyglot '''

import argparse
import torch.multiprocessing as mp

from src.utils import setup
from src.problyglot import Problyglot

parser = argparse.ArgumentParser(description="Parses config files passed in via CLI")
parser.add_argument("Path", metavar='path', type=str, help='path to the config file')
args = parser.parse_args()

# ENTRY POINT 
def main():
    config = setup(args.Path)
    problyglot = Problyglot(config)
    problyglot()

if __name__ == '__main__':
    main()



