__author__ = 'Richard Diehl Martinez'

'''
Entry point for launching problyglot.
'''

import logging
import torch
import click
import os

from configparser import ConfigParser
from src.utils import set_seed
from src.dataset import MetaDataset
from src.dataloader import MetaDataLoader
from src.problyglot import Problyglot

'''
Base utils for setting up training and eval loops.
'''

def setup_config(config_file_path):
    '''Reads in a config file using ConfigParser'''
    config = ConfigParser()
    config.read(config_file_path)
    return config

def setup_logger(config_file_path):
    '''Sets up logging functionality '''
    # Removing handlers that might be associated with environment; and logs
    # out to both stderr and a log file
    experiment_directory = os.path.dirname(os.path.join(os.getcwd(), config_file_path)) 

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    log_file_name = os.path.join(experiment_directory, "experiment.log")
    logging.basicConfig(
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        level=logging.DEBUG,
                        handlers=[
                            logging.FileHandler(log_file_name),
                            logging.StreamHandler()
                        ]
    )
    logging.info(f"Initializing experiment: {experiment_directory}")

def setup(config_file_path):
    ''' 
    Reads in config file, sets up logger and sets a seed to ensure reproducibility.
    '''
    config = setup_config(config_file_path)
    setup_logger(config_file_path)
    set_seed(config.getint("EXPERIMENT", "set_seed", fallback=42))
    return config

# Entry point
@click.command()
@click.argument('config_fp')
def main(config_fp):
    config = setup(config_fp)
    train_meta_dataset = MetaDataset(config, meta_split='train')
    train_meta_dataloader = MetaDataLoader(train_meta_dataset)
    problyglot = Problyglot(config)

    problyglot.train(train_meta_dataloader)

    train_meta_dataset.shutdown()

if __name__ == '__main__':
    main()