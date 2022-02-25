__author__ = 'Richard Diehl Martinez' 
""" Dataset class for iterating over XNLI data"""

import logging
import os

from torch.utils.data import IterableDataset
from transformers import XLMRobertaTokenizer

logger = logging.getLogger(__name__)

# to stop the huggingface tokenizer from giving the sequence longe than 512 warning 
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class XNLIDataset(IterableDataset):

    """ Dataset for the XNLI corpus (https://github.com/facebookresearch/XNLI) """

    def __init__(self, config): 
        """
        We assume that the data for the xnli task has been downloaded as part of the 
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme#download-the-data).

        Initialize a dataset for iterating over the XNLI via a config file.
        """
        # location of folder containing xnli data
        self.root_path = config.get("XNLI_DATASET", "root_path")

        self.use_few_shot_adaptation = config.getboolean("XNLI_DATASET", "use_few_shot_adaptation", fallback=False)

        if self.use_few_shot_adaptation:
            # TODO: Implement ability to do few-shot adaptation; where the finetune data is the same as the evaluation datas
            raise NotImplementedError()

    @staticmethod
    def _get_language_files(root_path):
        """ 
        Helper function for setting up finetune-evaluation data pairs.

        Returns: 
            * A list of dictionaries of the following format: 
                {"finetune": (finetune language (str), finetune data file path),
                "evaluation": (evaluation language (str), evaluation data file path),
                }

        """
        file_paths = os.listdir(root_path)

    
        for file_path in file_paths: 
            


    def __iter__(self): 
        """ IterableDataset expects __iter__ to be overriden"""
        pass

    def __next__(self):
        


def __main__():
    from configparser import ConfigParser

    config = ConfigParser()
    config.add_section('XNLI_DATASET')
    config.set('XNLI_DATASET', 'root_path', 'data/xtreme/download/xnli')

    print("Testing functionality of XNLIDataset class")

    print("Initializing dataset")

    dataset = XNLIDataset(config)


if __name__ == '__main__':
    main()

