__author__ = 'Richard Diehl Martinez'
''' General purpose dataloader for NLU datasets '''

import torch
from torch.utils.data import DataLoader

def nlu_collate(batch):
    """ 
    Process a batch of nlu task data.
    
    Args: 
        * batch: list of tuples where the first entry is a list of input token ids,
            and the second element is an integer representing the label id

    Returns: 
        * processed_batch: a dictionary containing the following information
            * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
            * input_target_idx (torch.tensor): Tensor indicating for each sample at what index we apply 
                the final classification layer 
            * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
            * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
    """

    batch_inputs = []
    batch_labels = []
    batch_max_seq_len = 0 

    for sample_input_ids, sample_label_id in batch:
        len_sample = len(sample_input_ids)
        if len_sample > batch_max_seq_len:
            batch_max_seq_len = len_sample
        
        batch_inputs.append(sample_input_ids)
        batch_labels.append(sample_label_id)
        
        
    # NOTE: Padding token needs to be 1, in order to be consistent with hugging face tokenizer: 
    # https://huggingface.co/transformers/model_doc/xlmroberta.html#transformers.XLMRobertaTokenizer
    input_tensor = torch.ones((len(batch), batch_max_seq_len))

    for idx, sample in enumerate(batch_inputs): 
        input_tensor[idx, :len(sample)] = torch.tensor(sample)

    # NOTE: the target idx is index over which we want to apply the classifier 
    # for NLU tasks we apply the classifier over the CLS token which is always at index 0
    input_target_idx = torch.zeros((len(batch)))
    label_tensor = torch.tensor(batch_labels)
    attention_mask_tensor = (input_tensor != 1)

    processed_batch = {
        "input_ids": input_tensor.long(),
        "input_target_idx": input_target_idx.long(),
        "label_ids": label_tensor.long(),
        "attention_mask": attention_mask_tensor.int()
    }

    return processed_batch


class NLUDataLoader(DataLoader):
    """ 
    Minimal wrapper around DataLoader to override the default collate_fn to be 
    nlu_collate.
    """
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=nlu_collate, **kwargs)