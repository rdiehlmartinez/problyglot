__author__ = 'Richard Diehl Martinez'

'''
Entry point for launching problyglot.
'''

import logging
import torch
import click
import os

from configparser import ConfigParser
from src.utils import device, set_seed
from src.dataset import MetaDataset
from src.dataloader import MetaDataLoader

'''
Base utils for setting up training and eval loops.
'''

def setup_config(config_file_path):
    config = ConfigParser()
    config.read(config_file_path)
    return config

def setup_logger(config_file_path):
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
    logging.info(f"Running model on device: {device}")

def setup(config_file_path):
    ''' '''
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

    for batch in train_meta_dataloader:
        batch_language, support_batch, query_batch = batch
        # Process batch
        # TODO 

    train_meta_dataset.shutdown()

if __name__ == '__main__':
    main()