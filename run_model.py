__author__ = 'Richard Diehl Martinez'
""" Entry point for launching problyglot """

import argparse
import signal

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

    # setting up timeout handler - called if the program receives a SIGINT 
    # either from the user or from SLURM if it is about to timeout
    signal.signal(signal.SIGINT, problyglot.timeout_handler)

    # launching training or eval script
    problyglot()

if __name__ == '__main__':
    main()



