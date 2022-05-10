__author__ = 'Richard Diehl Martinez'
""" Entry point for launching problyglot """

import argparse
import signal

import torch.multiprocessing as mp

from src.utils import setup
from src.problyglot import Problyglot

parser = argparse.ArgumentParser(description="Parses config files passed in via CLI")
parser.add_argument("Path", metavar='path', type=str, help='path to the config file')
parser.add_argument('--resume_run_id', type=str, help="""
(IMPORTANT: should not be called by user) - will be set by the program if it encounters 
a time expiration error that forces it to rerun this program which is defined by this value.
""")
parser.add_argument('--resume_num_task_batches', type=int, help="""
(IMPORTANT: should not be called by user) - will be set by the program if it encounters 
a time expiration error that forces it to rerun this program, in which case it will resume 
training from this specified value. 
""")
args = parser.parse_args()

# ENTRY POINT 
def main():
    config = setup(args.Path, resume_run_id=args.resume_run_id, 
                              resume_num_task_batches=args.resume_num_task_batches)
    
    problyglot = Problyglot(config, resume_run_config_path=args.Path,
                                    resume_run_id=args.resume_run_id, 
                                    resume_num_task_batches=args.resume_num_task_batches)

    # setting up timeout handler - called if the program receives a SIGINT 
    # either from the user or from SLURM if it is about to timeout
    signal.signal(signal.SIGINT, problyglot.timeout_handler)

    # launching training or eval script
    problyglot()

if __name__ == '__main__':
    main()



