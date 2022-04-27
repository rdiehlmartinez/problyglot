__author__ = 'Richard Diehl Martinez' 
""" Dataset class for iterating over XNLI data"""

import logging
import os

from collections import defaultdict

from transformers import XLMRobertaTokenizer

from .nludataset import NLUDatasetGenerator, NLUDataset

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')

# to stop the huggingface tokenizer from giving the sequence longer than 512 warning 
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

class XNLIDatasetGenerator(NLUDatasetGenerator):

    ''' 
    A generator that yields XNLIDataset classes and contains arguments for how to setup and run 
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

        # NOTE: If adapt_on_eval is True, then - assuming we are using platipus - we take 
        # an adaptation step on the final evaluation dataset. If we are not using platipus, 
        # we just ignore this flag
        self.adapt_on_eval = config.getboolean("XNLI", "adapt_on_eval", fallback=True)

        # how to initialize the XNLI (aka. classification) task head
        self.task_head_init_method = config.get("XNLI", "task_head_init_method", fallback="random")

        self.language_files = self._get_language_files(self.root_path)

        super().__init__(config)


    def _get_language_files(self, root_path):
        """ 
        Helper function for setting up finetune-evaluation data pairs.

        Returns: 
            * language_files: list of dictionaries of the following format: 
                {"finetune": (finetune language (str), finetune data file path),
                "evaluation": (evaluation language (str), evaluation data file path)}

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
                language_file_dict['finetune'] = {"lng": file_path_lng,
                                                  "file_path": translated_full_file_path}
            else: 
                # if we are doing zero-shot adaptation, then we always finetune on english data
                language_file_dict['finetune'] = {"lng": eng_lng_str, "file_path": eng_file_path}
            
            full_file_path = os.path.join(root_path, file_path)
            language_file_dict['evaluation'] = {"lng": file_path_lng, "file_path": full_file_path}

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
                                    and language_file_dict['finetune']['lng'] != "en"

            finetune_dataset = XNLIDataset(**language_file_dict['finetune'],
                                           language_task_kwargs=self.language_task_kwargs,
                                           translated=finetune_translated)
            evaluation_dataset = XNLIDataset(**language_file_dict['evaluation'],
                                             language_task_kwargs=self.language_task_kwargs)

            yield (finetune_dataset, evaluation_dataset)


class XNLIDataset(NLUDataset):

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

    def __init__(self, translated=False, **kwargs): 
        """
        For a given language string and data filepath, establishes an IterableDataset that 
        iterates over and processes the XNLI data for that language. The keyword arg, translated,
        indicates whether the data has been translated in which case the data preprocessing
        differs slightly.
        """

        super().__init__(**kwargs)
        self.translated = translated

    def preprocess_line(self, line, process_for_adaptation=False):
        """
        For a given text input line, splits the line into the hypothesis and the premise; and 
        tokenizes the two lines. If process_for_adaptation is set, returns a tuple of the two 
        tokenized lines. Otherwise, returns the two lines tokenized into one combined list of  
        tokens along with the corresponding label id.

        Args: 
            * line (str): Line of text 
            * process_for_adaptation (bool): Whether to process the line for generating a batch
                of data for model adaptation (only applicable if using platipus). 
            
        
        Returns: 
            If process_for_adaptation: 
                * Tuple of lists corresponding to tokenized hypothesis and premise 
            Else:
                * input_ids (list): List of tokens of combined hypothesis and premise
                * label_id (int): Label for the current sample
        """

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

        if process_for_adaptation: 
            text_a_token_ids = tokenizer.encode(text_a)
            text_b_token_ids = tokenizer.encode(text_b)
            return (text_a_token_ids, text_b_token_ids)
        else:
            # tokenizing inputs
            inputs = tokenizer.encode_plus(text_a, text_b, add_special_tokens=True,
                                           max_length=XNLIDataset.MAX_SEQ_LENGTH)
            
            input_ids = inputs['input_ids']
            label_id = XNLIDataset.LABEL_MAP[label]

            return [input_ids, label_id]

def main():
    """ Basic testing of XNLI Dataset"""
    from configparser import ConfigParser

    from nludataloader import NLUDataLoader

    config = ConfigParser()
    config.add_section('XNLI')
    config.set('XNLI', 'root_path', '../../data/xtreme/download/xnli')
    config.set('XNLI', 'use_few_shot_adaptation', 'True')


    config.add_section('LEARNER')
    config.set('LEARNER', 'method', 'platipus')

    config.add_section('LANGUAGE_TASK')
    config.set('LANGUAGE_TASK', 'n', '2')
    config.set('LANGUAGE_TASK', 'k', '2')
    config.set('LANGUAGE_TASK', 'q', '20')
    config.set('LANGUAGE_TASK', 'sample_size', '10_000')
    config.set('LANGUAGE_TASK', 'buffer_size', '100_000_000')
    config.set('LANGUAGE_TASK', 'mask_sampling_method', 'proportional')
    config.set('LANGUAGE_TASK', 'mask_sampling_prop_rate', '0.3')
    config.set('LANGUAGE_TASK', 'max_seq_len', '128')

    dataset_generator = XNLIDatasetGenerator(config)

    for finetune_dataset, evaluation_dataset in dataset_generator:
        
        if finetune_dataset.language == "en":
            adaptation_batch = finetune_dataset.get_adaptation_batch()
            print(adaptation_batch)

if __name__ == '__main__':
    main()

