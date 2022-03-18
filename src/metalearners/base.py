__author__ = 'Richard Diehl Martinez'
''' Base ABC Class for (meta) learners '''

import abc 
import torch

class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        """
        BaseLearner establishes the inferface for the learner class and 
        inherents from torch.nn.Module
        """
        super().__init__()

    # Methods for training model 

    @abc.abstractmethod
    def optimizer_step(self, set_zero_grad=False):
        """ 
        Take a global update step of the meta learner params; optionally set the 
        gradients of the meta learner gradient tape back to zero 
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inner_loop(self, support_batch, query_batch=None, *args, **kwargs): 
        """ 
        Run an inner loop optimization step (in the context of meta learning); assumes 
        that the class contains the model that is to-be meta-learned.

        Args:
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index we apply 
                    the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
            * query_batch [optional]: same as support_batch, but for the data of the query set. This
                argument might be optional depending on how the learner is implemented.

        Returns: 
            * loss (torch.tensor): a tensor containing the loss that results from the inner loop 
        """
        raise NotImplementedError()

    # Methods for evaluating model 

    @abc.abstractmethod
    def run_finetuning_classification(self, finetune_dataloader, *args, **kwargs):
        """
        Args:
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model is passed
                in as a dataloader (in most cases this will be an NLUDataloader)

        Returns:
            * inference_params ({}): Returns any sort of params that need to be used for the 
                run_inference_classification method (e.g. finetuned weights of the model for platipus)
        """ 
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inference_classification(self, inference_dataloader, *args, **kwargs):
        """
        Args: 
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)

        Returns: 
            * predictions ([int]): a dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): the value of the classification loss on the inference dataset.
        """
        raise NotImplementedError()