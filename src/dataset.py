__author__ = 'Richard Diehl Martinez'
'''Utilities for preprocessing and loading dataset '''

import os
import torch 
import gzip
import multiprocessing
import mmap
import random
import time
import logging
import math
import numpy as np

from transformers import XLMRobertaTokenizer
from collections import defaultdict

logger = logging.getLogger(__name__)

# We always use the XLM sentencepiece tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
MASK_TOKEN_ID = tokenizer.mask_token_id 
# to encode any token id we need BYTE_ENCODING_SIZE number of bytes
BYTE_ENCODING_SIZE = math.ceil(math.log(tokenizer.vocab_size + 1, 16)) 
BYTE_ENDIAN_MODE = 'big'
BYTE_END_MARKER = tokenizer.vocab_size.to_bytes(BYTE_ENCODING_SIZE, BYTE_ENDIAN_MODE)

class MetaDataset(object):
    """
    MetaDataset that coordinates the generation of datasets for meta-train, -dev and -test.
    """

    '''
    * Needs to keep track of which words have been included in train/dev/test split 
    * Initializes BaseDataset datasets that on initialization spin up workers to read in data to a data buffer
        * there sort into train, eval, test based on some rules 
    * Implements a sampling procedure to sample the next language 
    '''

    @staticmethod
    def _initialize_datasets(languages, data_root, return_metadata=True):
        """ 
        Helper method for setting up datasets 
        Args: 
            * languages [List]: list of languages stored as iso-codes 
            * data_root (str): File path of root of data directory
            * [optional] return_metadata (bool): whether to return meta-data associated with the data,
                such as the amount of data available for a given language 
        Returns:
            * datasets {lng: BaseIterableDataset}: Returns a dictionary mapping a specific 
                language to the associated dataset for that language 
            * datasets {lng: dict()}: Returns a dictionary mapping a specific language to 
                metadata associated with the dataset for that language

        """

        def compute_dataset_size(lng_root_fp): 
            """ Calculate the size of a directory in bytes"""
            size = 0 
            for filename in os.listdir(lng_root_fp):
                filepath = os.path.join(lng_root_fp, filename)
                size += os.stat(filepath).st_size
            return size

        datasets = {}
        datasets_md = {}
        for language in languages: 
            lng_root_fp = os.path.join(data_root, language)

            dataset_size = compute_dataset_size(lng_root_fp)

            dataset = IterableLanguageTaskDataset(lng_root_fp, language, cycle_data=True)
            
            datasets[language] = dataset
            datasets_md[language] = {"dataset_size": dataset_size} # Can add more metadata 

            # NOTE: Test code
            num_sec_wait = 10
            for n in range(4): 
                print(f"Main loop: starting iteration {n}")
                print(f"Main loop: Doing some work for {num_sec_wait} seconds")
                time.sleep(num_sec_wait)
                print(f"Main loop: Calling next(dataset)")
                curr_batch = next(dataset)
                print(f"Main loop: Batch of data")
                print(curr_batch)

            dataset.shutdown()
            exit()

        return datasets, datasets_md 

    @staticmethod
    def _get_languages(config): 
        """ Helper for reading in languages from config or from a file """    
        processed_languages = []    
        for partition in ["train", "dev", "test"]: 
            # partition_languages_str can either be empty string, a file name or a 
            # comma-separated list of iso language codes 
            partition_languages_str = config.get("DATA", f"{partition}_languages", fallback="")
            if ".txt" in partition_languages_str: 
                with open(f"configs/demo/{partition_languages_str}") as f: 
                            partition_languages = f.read().splitlines()
                            processed_languages.append(partition_languages)
            elif partition_languages_str == "":
                assert(partition != "train"), "config item for train_languages cannot be empty"
                processed_languages.append(None)
            else: 
                partition_languages = partition_languages_str.split(",")
                processed_languages.append(partition_languages)
        
        return processed_languages


    def __init__(self, config):
        """ 
        Initialize datasets for meta-train, -dev and -test
        """

        train_languages, dev_languages, test_languages =  self._get_languages(config)

        data_root = config.get("DATA", "root_path")
        self.train_datasets, self.train_datasets_md = self._initialize_datasets(train_languages, data_root)

    def get_dataset(split): 
        """ 
        For a given split ('train', 'dev', 'test) generates a dataset for either training or validation
        """
        raise NotImplementedError()


class IterableLanguageTaskDataset(object): 
    ''' 
    Iterable dataset that reads language data from a provided directory of txt files 
    and returns at each iteration some N-way K-shot example
    '''
    def __init__(self, root_fp, lng, 
                                N = 10,
                                K = 5,
                                Q = 10,
                                buffer_size=1e6,
                                sample_size=100_000,
                                sampling_method="random",
                                cycle_data=True,
                                **kwargs): 
        """ 
        Initializes params and data buffers for the iterable dataset. 

        For each iteration reads in (sample_size) sentences from the dataset, and from those 
        sentences samples a N-way K-shot 'task'. This process happens on a worker node, which 
        then communicates with the parent node by writting out the N-way K-shot data to a 
        data buffer that is stored in shared memory. 

        Note then that when the parent calls the get_stream() method, we only need to 
        read from the data buffer which can happen quickly.
        
        Args: 
            * root_fp (str): A file path to the directory where the language data is 
                stored. This directory can have many files which are all 
                expected to be .txt.gz files, where each line is a new sample.
            * lng (str): iso code for the language corresponding to the dataset
            * [optional] N (int): The N in N-way K-shot classification (defaults to 10)
            * [optional] K (int): The K in N-way K-shot classification (defaults to 5)
            * [optional] Q (int: The number of samples in the query set for each 'task' (defaults to 10)
                Thus for each class, we must find K+Q examples of that class
            * [optional] buffer_size (int): size of the memory-mapped buffer (defaults to 100,000 bytes)
            * [optional] sample_size (int): number of phrases to sample before returning a sample for 
                N-way k-shot classification (defaults to 100,000)
            * [optional] sampling_method (str): either one of 'random' or 'proportional' which specify 
                how to sample the N tasks
            * [optional] cycle_data (bool): whether to continually cycle over the data (defaults to true)
        """
        super().__init__()
        self.root_fp = root_fp 
        self._lng = lng
        self.N = 3 #N 
        self.K = 2 #K
        self.Q = Q

        # NOTE: Each sample requires roughly 1000 bytes to store (~liberal heuristic)
        if (N*K*1000 > buffer_size): 
            logger.warning(f"The buffer size used in BaseIterableDataset ({buffer_size} bytes) is likely too small")

        self.sample_size = 10_000 #sample_size 
        self.sampling_method = sampling_method
        self.cycle_data = cycle_data

        # event and lock to communicate between parent and child 
        self.event = multiprocessing.Event()
        self.lock = multiprocessing.Lock()

        # Extract data out of the buffers for train and dev (i.e. support and query)
        self.train_data_buffer = mmap.mmap(-1, length=int(buffer_size))
        self.dev_data_buffer = mmap.mmap(-1, length=int(buffer_size))
        
        self.worker = multiprocessing.Process(
            target=self.generate_buffer,
        )

        self.worker.start()

    @property
    def language(self):
        """ Language property """
        return self._lng

    def shutdown(self): 
        """ Needs to be called in order to terminate the data generation worker """
        self.worker.terminate()
        self.worker.join()

    # --- The following methods should only be called by the child process ---
    
    @staticmethod
    def _tokenize_line(raw_line):
        """ Decode and tokenize a raw text string """
        decoded_line = raw_line.decode('utf-8')
        tokenized_line = tokenizer(decoded_line)
        input_ids = tokenized_line['input_ids']
        return input_ids
    
    @staticmethod
    def _process_file_paths(root_fp):
        """ Filters and shuffles the file paths stored in self.fp """
        file_paths = os.listdir(root_fp)

        # ensure all files are text files - otherwise no guarantees
        file_paths = list(filter(lambda x: ".txt" in x, file_paths))
        random.shuffle(file_paths)

        return file_paths

    def generate_N_K_samples(self, curr_subword_to_sample, curr_samples):
        """
        Given a set of samples (curr_samples) drawn from the dataset generates a 
        sample for N-way K-shot classification + Q samples for the query set 

        Args: 
            * curr_subword_to_sample {subword_token: [(index of occurence in curr_samples,
                                                       index of occurence within a sample)]:
                A dictionary mapping a current subword token to a tuple containing: 
                1. the index of the sample in curr_samples where that subword token occurs and 
                2. within the given sample the index of the location where that token occurs
                We do this because a given token can occur multiple times within a given sample
            * curr_samples [List]: A list of self.sample_size number of samples 

        Returns: 
            * support_set {token id: [K samples where token id occurs]}: mapping of N token ids 
                to K samples per token id occurs
            * query_set {token id: [Q samples where token id occurs]}: mapping of N token ids 
                to Q samples per token id occurs

        """
    
        support_set = defaultdict(list)
        query_set = defaultdict(list)

        # Filter out words that do not occur K times 
        filtered_subword_to_sample = {
            k: v for k,v in curr_subword_to_sample.items() if len(v) >= (self.K + self.Q)
        }

        # Checking whether the dataset is too small
        assert(len(filtered_subword_to_sample) > self.N),\
            f"Not enough data to generate N-way k-shot samples for dataset: {self.langauge}"

        # sampling mechanism for getting the N classes
        if self.sampling_method == 'random': 
            sampled_N = random.sample(filtered_subword_to_sample.keys(), k=self.N)
        elif self.sampling_method == 'proportional':
            # TODO
            # random.choices
            # sampling_weights = [len(v) for v in filtered_subword_to_sample.values()]
            logger.error("Sampling method: proportional has not been implemented yet")
            raise NotImplementedError()
        else: 
            logger.error(f"Invalid sampling method: {self.sampling_method}")
            raise Exception(f"Invalid sampling method: {self.sampling_method}")


        def mask_sample(k_index_information):
            """
            Given k_index_information, a tuple containing the following info,
                1. the index in curr_samples where the subword occurs
                2. within the sample the index where the sample occurs 

            returns a sample with the correct subword masked out.
            """
           
            across_sample_index, within_sample_index = k_index_information
            curr_sample = curr_samples[across_sample_index]
            curr_sample[within_sample_index] = MASK_TOKEN_ID

            return curr_sample
    
        # now sample the k+q samples for each class
        for sampled_n in sampled_N: 
            # for each class i.e. n in {1,..., N} generate k sentences randomly
            # note that in a given sample there might be multiple occurences of a token
            # so we need to specify which token it is we want to 
            subword_indices = filtered_subword_to_sample[sampled_n]

            sampled_K_plus_Q_indices = random.sample(subword_indices, k=(self.K + self.Q))

            for k_index_information in sampled_K_plus_Q_indices[:self.K]:
                masked_sample = mask_sample(k_index_information)
                support_set[sampled_n].append(masked_sample)

            for q_index_information in sampled_K_plus_Q_indices[self.K:]:
                masked_sample = mask_sample(q_index_information)
                query_set[sampled_n].append(masked_sample)

        return (support_set, query_set)

    @staticmethod
    def write_to_buffer(curr_set, curr_buffer):
        """ For the support and query set, write the data out to the respective buffers """
        for subword_id, samples in curr_set.items():
            curr_buffer.write(subword_id.to_bytes(BYTE_ENCODING_SIZE, BYTE_ENDIAN_MODE))
            curr_buffer.write(BYTE_END_MARKER)
            for sample in samples:
                for sample_tok_id in sample:
                    encoded_token_id = sample_tok_id.to_bytes(BYTE_ENCODING_SIZE, BYTE_ENDIAN_MODE)
                    curr_buffer.write(encoded_token_id)
                curr_buffer.write(BYTE_END_MARKER)
            
        curr_buffer.flush()

    def release_and_wait(self):
        """
        NOTE: This should only ever be run by a child worker.

        Helper function for releasing a lock and waiting to reacquire the lock
        to begin writing to buffer again.  
        """
        print("In Child: Finished writing data")
        print("In Child: calling release_and_wait")
        self.lock.release()
        self.event.clear()
        print("In Child: waiting for event")
        self.event.wait()
        self.lock.acquire() 
        print("In Child: Starting to write fresh batch to buffer")
        self.train_data_buffer.seek(0) 
        self.dev_data_buffer.seek(0)

    def generate_buffer(self):
        """ 
        NOTE: This should only ever be run by a child worker. 
        This method generates a stream of data that is stored in a buffer 
        from where it can be accessed by the parent process to generate train 
        and val data. 

        This method will loop forever if self.cycle_data is set to true. 
        We usually want to randomly and forever loop over the data so we cycle over 
        the file paths indefinitely - the worker only stops when it is shut down by
        the main process.
        """

        # This lock is acquired when worker is initially launched
        self.lock.acquire()
        print("In Child: Beginning generate_buffer function")

        # keeps track of edge case where the entire dataset is smaller than self.sample_size
        is_too_small = False 
        total_samples_processed = 0 
        
        curr_samples_processed = 0 
        curr_samples = []
        curr_subword_to_sample = defaultdict(list)

        file_paths = self._process_file_paths(self.root_fp)

        while True:
            for curr_fp in file_paths:

                with gzip.open(os.path.join(self.root_fp, curr_fp)) as gzip_buffer: 
                    for curr_line in gzip_buffer:

                        if curr_samples_processed < self.sample_size: 
                            token_ids = self._tokenize_line(curr_line)

                            if len(token_ids) > tokenizer.max_len_single_sentence:
                                # skip the current sample if it is too large for the model
                                continue

                            curr_samples.append(token_ids)
                            for idx, token_id in enumerate(token_ids):
                                if token_id in tokenizer.all_special_ids:
                                    # don't include special tokens 
                                    continue
                                curr_subword_to_sample[token_id].append((curr_samples_processed, idx))
                            
                            curr_samples_processed += 1
                            total_samples_processed += 1 

                        if curr_samples_processed == self.sample_size: 
                            # done reading in all of the data 

                            support_set, query_set = self.generate_N_K_samples(curr_subword_to_sample, curr_samples)

                            # writing data out to buffer 
                            try:
                                self.write_to_buffer(support_set, self.train_data_buffer)
                                self.write_to_buffer(query_set, self.dev_data_buffer)
                            except ValueError as e: 
                                raise Exception(f"Buffer for dataset: {self.language} ran out of space")

                            # resetting per-sample data structures 
                            curr_samples_processed = 0 
                            curr_samples = []
                            curr_subword_to_sample = defaultdict(list)
                            
                            self.release_and_wait()

            # NOTE: Just finished looping over the entire dataset 
            
            if total_samples_processed < self.sample_size: 
                # will possibly trigger after first pass through the entire dataset 
                logger.warning(f"Size of dataset for language {self.language}: {total_samples_processed} is smaller than {self.sample_size} samples")
                is_too_small = True

            if is_too_small: 
                # we have looped over entire dataset before sampling sample_size samples

                support_set, query_set = self.generate_N_K_samples(curr_subword_to_sample, curr_samples)

                # writing data out to buffer 
                try:
                    self.write_to_buffer(support_set, self.train_data_buffer)
                    self.write_to_buffer(query_set, self.dev_data_buffer)
                except ValueError as e: 
                    raise Exception(f"Buffer for dataset: {self.language} ran out of space")

                # resetting per-sample data structures 
                curr_samples_processed = 0 
                curr_samples = []
                curr_subword_to_sample = defaultdict(list)
                
                self.release_and_wait()

            if not self.cycle_data:
                self.lock.release()
                break

    def __next__(self): 
        """ 
        NOTE: Called from main process
        Reads and returns the data that has been stored in the train_data_buffer and the 
        dev_data_buffer by the worker node.

        Returns: 
            * train_samples {token_id : [K samples of token_id masked out]}: Mapping of 
                N different token_ids to K samples of sentences where the token is masked out.
            * dev_samples {token_id : [Q samples of token_id masked out]}: Mapping of 
                N different token_ids to Q samples of sentences where the token is masked out.
        """
        print("In Main: Calling Next()")
        self.lock.acquire()
        print("In Main: Lock acquired and returning processed buffer")

        self.train_data_buffer.seek(0)
        self.dev_data_buffer.seek(0)

        train_samples = defaultdict(list)
        dev_samples = defaultdict(list)

        for return_dict, data_buffer, num_samples_per_n in [(train_samples, self.train_data_buffer, self.K),
                                                            (dev_samples, self.dev_data_buffer, self.Q)]:
            for n in range(self.N): 

                curr_n = int.from_bytes(data_buffer.read(BYTE_ENCODING_SIZE), BYTE_ENDIAN_MODE)

                # If the bytes following the initial token_id are not the end_marker then
                # buffer state is wrong 
                assert(data_buffer.read(BYTE_ENCODING_SIZE) == BYTE_END_MARKER)

                for k in range(num_samples_per_n):
                    curr_sample = []
                    while True: 
                        curr_encoded_token = data_buffer.read(BYTE_ENCODING_SIZE)
                        if (curr_encoded_token == BYTE_END_MARKER):
                            break
                        curr_token = int.from_bytes(curr_encoded_token, BYTE_ENDIAN_MODE)
                        curr_sample.append(curr_token)
                    return_dict[curr_n].append(curr_sample)

        print("In Main: finished next() - releasing lock and setting event")
        self.lock.release()
        self.event.set()
        return (train_samples, dev_samples)

    def __iter__(self):
        """ To comply with iterator protocol """
        return self
