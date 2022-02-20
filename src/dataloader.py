__author__ = 'Richard Diehl Martinez '
'''Utilities for dataloading datasets'''

import torch

from torch.utils.data import DataLoader
from .dataset import MASK_TOKEN_ID, CLS_TOKEN_ID

def meta_collate(batch):
    """ 
    Transform a batch of task data into input and label tensors that can be fed 
    into a model. 
    
    Args: 
        * batch: tuple containing the following: 
            * task_name (str): task name (e.g. the language) of the current batch of data 
            * (support_set, query_set), where this tuple contains the following: 
                * support_set {token id: [K samples where token id occurs]}: mapping of N token ids 
                    to K samples per token id occurs
                * query_set {token id: [Q samples where token id occurs]}: mapping of N token ids 
                    to Q samples per token id occurs

    Returns: 
        * task_name (str): task name (e.g. the language) of batch
        * support_batch: a dictionary containing the following information for the support set
            * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
            * input_target_idx (torch.tensor): Tensor indicating for each sample at what index we apply 
                the final classification layer 
            * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
            * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
        * query_batch: same as support_batch, but for the data of the query set 

    """
    task_name, (support_samples, query_samples) = batch[0] # only 1-task per batch 

    # if the task is an nlu task we classify over the CLS token, otherwise over the MASK token
    if 'nlu' in task_name:
        target_tok_id = CLS_TOKEN_ID
    else:
        target_tok_id = MASK_TOKEN_ID

    def process_batch(batch_samples):
        """ 
        Helper function to process samples from either the support or query sets,
        and return a dictionary of input_ids, input_target_idx, label_ids and attention_mask
        (i.e. the expected data structure returned by meta_collate).
        """

        batch_inputs = []
        batch_input_target_idx = []
        batch_labels = []
        batch_max_seq_len = 0 

        for idx, (_, subword_samples) in enumerate(batch_samples.items()):
            # randomly assigns each subword_idx to a number in range(N)
            batch_labels.append(idx) 
            for subword_sample in subword_samples:
                if len(subword_sample) > batch_max_seq_len:
                    batch_max_seq_len = len(subword_sample)
                batch_inputs.append(subword_sample)
            
        # NOTE: Padding token needs to be 1, in order to be consistent with hugging face tokenizer: 
        # https://huggingface.co/transformers/model_doc/xlmroberta.html#transformers.XLMRobertaTokenizer
        input_tensor = torch.ones((len(batch_inputs), batch_max_seq_len))

        for idx, sample in enumerate(batch_inputs): 
            input_tensor[idx, :len(sample)] = torch.tensor(sample)
            batch_input_target_idx.append(sample.index(target_tok_id))

        input_target_idx = torch.tensor(batch_input_target_idx)
        label_tensor = torch.tensor(batch_labels)
        attention_mask_tensor = (input_tensor != 1)

        processed_batch = {
            "input_ids": input_tensor.long(),
            "input_target_idx": input_target_idx.long(),
            "label_ids": label_tensor.long(),
            "attention_mask": attention_mask_tensor.int()
        }

        return processed_batch
    
    support_batch = process_batch(support_samples)
    query_batch = process_batch(query_samples)

    return (task_name, support_batch, query_batch)
    

class MetaDataLoader(DataLoader):
    """
    Stripped down basic dataloader meant to be used with MetaDataset,
    note that MetaDataset does most of the heavy-lifting with processing 
    the data. 
    
    Copied from: 
    https://github.com/tristandeleu/pytorch-meta/blob/master/torchmeta/utils/data/dataloader.py#L32
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                batch_sampler=None, num_workers=0, pin_memory=False,
                drop_last=False, timeout=0, worker_init_fn=None):
                
        """ Resetting basic defaults  """
        super().__init__(dataset, batch_size=batch_size,
            shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler,
            num_workers=num_workers, collate_fn=meta_collate,
            pin_memory=pin_memory, drop_last=drop_last, timeout=timeout,
            worker_init_fn=worker_init_fn)