__author__ = 'Richard Diehl Martinez' 
""" Dataset class for iterating over XNLI data"""

import logging
import os

from torch.utils.data import IterableDataset
from transformers import XLMRobertaTokenizer

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# to stop the huggingface tokenizer from giving the sequence longer than 512 warning 
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class XNLIDatasetGenerator():

    ''' 
    A generator that yields XNLIDataset classes and contains arguments for how to run 
    evaluation of the XNLI task.
    '''

    def __init__(self, config): 
        """
        We assume that the data for the xnli task has been downloaded as part of the 
        XTREME cross-lingual benchmark (https://github.com/google-research/xtreme).

        Initializes a generator class that yield at each iteration a tuple of 
        Dataset objects each representing a language in the XNLI corpus which are
        the data for (respectively) finetuning and evaluating the model on.
        """
        # location of folder containing xnli data
        self.root_path = config.get("XNLI", "root_path")

        # whether the evaluation is going to be done on dev or on test
        self.evaluation_partition = config.get("EVALUATION", "partition", fallback="dev")

        # if use_few_shot_adaptation is False then we do zero-shot adaptation
        # typically from english -> other language
        self.use_few_shot_adaptation = config.getboolean("XNLI", "use_few_shot_adaptation",
                                                         fallback=False)

        if self.use_few_shot_adaptation:
            self.translated_root_path = os.path.join(self.root_path, "translate-train")
            assert(os.path.exists(self.translated_root_path)),\
                "For few shot adaptation must have a translate-train directory"

        # if adapt_on_eval is True, then - assuming we are using platipus - we take 
        # an adaptation step on the final evaluation dataset.  
        self.adapt_on_eval = config.getboolean("XNLI", "adapt_on_eval", fallback=True)

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

        eng_lng_str = 'en'
        eng_file_path = os.path.join(root_path, 'train-en.tsv')

        if self.use_few_shot_adaptation:
            translated_file_paths = os.listdir(self.translated_root_path)

        for file_path in file_paths: 
            file_path_split = file_path.split('-')
            file_path_partition = file_path_split[0]
            file_path_lng = file_path_split[1].split('.')[0] # removing .tsv 

            if file_path_partition != self.evaluation_partition:
                continue

            language_file_dict = dict()

            if self.use_few_shot_adaptation and file_path_lng != "en":
                # looking up the translated version of the current evaluation file
                # except when the eval language is already english 
                translated_file_path = list(filter(lambda x: file_path_lng in x,
                                                   translated_file_paths))[0]
                translated_full_file_path = os.path.join(self.translated_root_path,
                                                         translated_file_path)
                language_file_dict['finetune'] = (file_path_lng, translated_full_file_path)
            else: 
                # if we are doing zero-shot adaptation, then we always finetune on english data
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
            # the finetuning set is translated if few shot learning is set and the current language 
            # is not english
            finetune_translated = self.use_few_shot_adaptation\
                                    and language_file_dict['finetune'][0] != "en"

            finetune_dataset = XNLIDataset(*language_file_dict['finetune'],
                                           translated=finetune_translated)
            evaluation_dataset = XNLIDataset(*language_file_dict['evaluation'])

            yield (finetune_dataset, evaluation_dataset)


class XNLIDataset(IterableDataset):

    # default value for XNLI 
    MAX_SEQ_LENGTH = 128

    # xnli classes
    LABEL_MAP = {"contradiction":0,
                 "entailment":1,
                 "neutral":2}

    """
    Dataset for processing data for a specific language in the XNLI corpus. 
    For batching, XNLIDataset expects to use an NLUDataLoader.
    """

    def __init__(self, lng, file_path, translated=False): 
        """
        For a given language string and data filepath, creates a Dataset that 
        iterates over and preprocesses the XNLI data for that language. 
        The keyword arg, translated, indicates whether the data has been 
        translated in which case the data preprocessing differs slightly.  
        """
        self._lng = lng
        self.file_path = file_path

        self.translated = translated

    @property
    def language(self):
        return self._lng

    def preprocess_line(self, line):
        # splitting information from tsv
        split_line = line.split('\t')

        if self.translated:
            text_a = split_line[2]
            text_b = split_line[3]
            label = split_line[4].strip()
            if label == 'contradictory':
                label = 'contradiction'
        else:
            text_a = split_line[0]
            text_b = split_line[1]
            label = split_line[2].strip()

        # tokenizing inputs
        inputs = tokenizer.encode_plus(text_a,
                                       text_b,
                                       add_special_tokens=True,
                                       max_length=XNLIDataset.MAX_SEQ_LENGTH)
        
        input_ids = inputs['input_ids']
        label_id = XNLIDataset.LABEL_MAP[label]

        return [input_ids, label_id]

    def __iter__(self): 
        """ IterableDataset expects __iter__ to be overriden"""
        with open(self.file_path, 'r') as f:
            for line in f:
                # tokenize line 
                processed_line = self.preprocess_line(line)
                yield processed_line

def main():
    """ Basic testing of XNLI Dataset"""
    from configparser import ConfigParser

    from nludataloader import NLUDataLoader

    config = ConfigParser()
    config.add_section('XNLI')
    config.set('XNL', 'root_path', '../../data/xtreme/download/xnli')
    config.set('XNLI', 'use_few_shot_adaptation', 'True')

    dataset_generator = XNLIDatasetGenerator(config)

    for finetune_dataset, evaluation_dataset in dataset_generator:

        finetune_dataloader = NLUDataLoader(finetune_dataset, batch_size=1280) 
        evaluation_dataloader = NLUDataLoader(evaluation_dataset, batch_size=3)

if __name__ == '__main__':
    main()

