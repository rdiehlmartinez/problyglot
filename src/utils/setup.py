__author__ = 'Richard Diehl Martinez'
''' Utilities for setting up logging and reading configuration files'''

import os
import logging 
import torch 
import numpy as np
import random

from configparser import ConfigParser

def set_seed(seed=42):
    ''' Sets seed for reproducibility '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(seed)

def setup_config(config_file_path):
    ''' Reads in a config file using ConfigParser '''
    config = ConfigParser()
    config.read(config_file_path)
    return config

def setup_logger(config_file_path):
    ''' Sets up logging functionality '''
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