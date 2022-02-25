__author__ = 'Richard Diehl Martinez' 
""" Dataset class for iterating over XNLI data"""

import logging
import os

from torch.utils.data import IterableDataset
from transformers import XLMRobertaTokenizer

logger = logging.getLogger(__name__)

# to stop the huggingface tokenizer from giving the sequence longe than 512 warning 
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class XNLIDatasetGenerator():

    """ Dataset for the XNLI corpus (https://github.com/facebookresearch/XNLI) """

    def __init__(self, config, evaluation_partition='dev'): 
        """
        We assume that the data for the xnli task has been downloaded as part of the 
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme#download-the-data).

        Initializes a generator class that yield at each iteration a tuple of 
        Dataset objects each representing a language in the XNLI corpus which are
        the data for (respectively) finetuning and evaluating the model on.
        """
        # location of folder containing xnli data
        self.root_path = config.get("XNLI_DATASET", "root_path")

        # whether the evaluation is going to be done on dev or on test
        self.evaluation_partition = evaluation_partition

        # if use_few_shot_adaptation is False then we do zero-shot adaptation (from english -> other language)
        self.use_few_shot_adaptation = config.getboolean("XNLI_DATASET", "use_few_shot_adaptation", fallback=False)

        if self.use_few_shot_adaptation:
            # TODO
            raise NotImplementedError()

        self.language_files = self._get_language_files(self.root_path)

    def _get_language_files(self, root_path):
        """ 
        Helper function for setting up finetune-evaluation data pairs.

        Returns: 
            * language_files: list of dictionaries of the following format: 
                {"finetune": (finetune language (str), finetune data file path),
                "evaluation": (evaluation language (str), evaluation data file path),
                }

        """
        language_files = []
        file_paths = os.listdir(root_path)

        if not self.use_few_shot_adaptation:
            # if we are doing zero-shot adaptation, then we always finetune on english data
            eng_lng_str = 'en'
            eng_file_path = os.path.join(root_path, 'train-en.tsv')
        

        for file_path in file_paths: 
            file_path_split = file_path.split('-')
            file_path_partition = file_path_split[0]
            file_path_lng = file_path_split[1].split('.')[0] # removing .tsv 

            if file_path_partition != self.evaluation_partition:
                continue

            language_file_dict = dict()

            if self.use_few_shot_adaptation:
                # TODO
                raise NotImplementedError()
            else: 
                language_file_dict['finetune'] = (eng_lng_str, eng_file_path)
            
            full_file_path = os.path.join(root_path, file_path)
            language_file_dict['evaluation'] = (file_path_lng, full_file_path)

            language_files.append(language_file_dict)
            
        return language_files

    def __iter__(self):
        """ 
        At each iteration yields XNLIDatasets which are the data for (respectively)
        finetuning and evaluating the model on.
        """

        for language_file_dict in self.language_files:
            finetune_dataset = XNLIDataset(*language_file_dict['finetune'])
            evaluation_dataset = XNLIDataset(*language_file_dict['evaluation'])

            yield (finetune_dataset, evaluation_dataset)


class XNLIDataset(IterableDataset):

    """ Dataset for processing data for a specific language in the XNLI corpus """

    def __init__(self, lng, file_path): 
        """
        For a given language string and data filepath, creates a Dataset that 
        iterates over and preprocesses the XNLI data for that language 
        """
        self._lng = lng
        self.file_path = file_path

    @property
    def language(self):
        return self._lng

    def __iter__(self): 
        """ IterableDataset expects __iter__ to be overriden"""
        with open(self.file_path, 'r') as f:
            for line in f:
                # tokenize line 
                yield line

def main():
    from configparser import ConfigParser

    config = ConfigParser()
    config.add_section('XNLI_DATASET')
    config.set('XNLI_DATASET', 'root_path', '../../data/xtreme/download/xnli')
    config.set('XNLI_DATASET', 'use_few_shot_adaptation', 'False')

    print("Testing functionality of XNLIDataset class")

    print("Initializing dataset")

    dataset_generator = XNLIDatasetGenerator(config, evaluation_partition='dev')

    for finetune_dataset, evaluation_dataset in dataset_generator:

        print(finetune_dataset.language)
        print(evaluation_dataset.language)

        for line in finetune_dataset: 
            print(line)
            break

                
        for line in evaluation_dataset: 
            print(line)
            break



if __name__ == '__main__':
    main()
