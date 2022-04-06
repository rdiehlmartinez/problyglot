__author__ = 'Richard Diehl Martinez'
''' Base ABC Class for (meta) learners '''

import abc 
import torch
import math

from ..taskheads import ClassificationHead

class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, base_model, device, **kwargs):
        """
        BaseLearner establishes the inferface for the learner class and 
        inherents from torch.nn.Module. 
        
        Args: 
            * base_model (implements a BaseModel)
            * device (str): what device to use for training (either 'cpu' or 'gpu) 
        """
        super().__init__()

        # hidden dimensions of the outputs of the base_model
        self.base_model_hidden_dim = base_model.hidden_dim 

        self.device = device

    ###### Task head initialization methods ######

    def get_task_init_kwargs(self, n_classes, **kwargs):
        """ 
        Helper method for generating keyword arguments that can be passed into a task head 
        initialization method
        
        Returns:
            * init_kwargs (dict): Keyword arguments used by the task head initialization function 
        """

        # The two attributes below need to be specified by the learner
        assert(hasattr(self, 'base_model_hidden_dim') and hasattr(self, 'device'))

        init_kwargs = {}

        init_kwargs['base_model_hidden_dim'] = self.base_model_hidden_dim
        init_kwargs['n_classes'] = n_classes
        init_kwargs['device'] = self.device

        return init_kwargs

    ###### Model training and evaluation helper methods ######

    @staticmethod
    def _compute_classification_loss(model_outputs, data_batch, task_head_weights):
        """
        Helper function for computing the classification loss on a given batch of data. We 
        assume that the data has already been passed through the base_model - the result of which
        is model_outputs (i.e. the final layer's hidden states). 

        Args: 
            * model_outputs (torch.Tensor): Result of passing data_batch through the 
                base_model. Should have shape: (batch_size, sequence_length, hidden_size)
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * task_head_weights (dict): weights used by the task head (in this the classifier head)
        Returns: 
            * logits ([torch.Tensor]): Logits resulting from forward pass 
            * loss (int): Loss of data 
        """
        # TODO: instead of having a _compute_x_loss for different classification heads
        # just pass in a keyword arg indicating what type of loss to use

        #indexing into sequence layer of model_outputs -> (batch_size, hidden_size) 
        batch_size = model_outputs.size(0)
        last_hidden_state = model_outputs[torch.arange(batch_size),
                                              data_batch['input_target_idx']]

        head = ClassificationHead()
        logits, loss = head(model_output=last_hidden_state, labels=data_batch['label_ids'],
                            weights=task_head_weights)

        return (logits, loss)

    ###### Model training methods ######

    def optimizer_step(self, set_zero_grad=False):
        """ 
        Take a global update step of the meta learner params; optionally set the 
        gradients of the meta learner gradient tape back to zero 
        """
        assert(hasattr(self, 'optimizer')),\
            "Learner cannot take optimizer step - needs to define an optimizer attribute"

        self.optimizer.step()
        if set_zero_grad:
            self.optimizer.zero_grad()

    @abc.abstractmethod
    def run_inner_loop(self, support_batch, query_batch=None, *args, **kwargs): 
        """ 
        Run an inner loop optimization step (in the context of meta learning); assumes 
        that the class contains the model that is to-be meta-learned.

        Args:
            * support_batch: A dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch [optional]: Same as support_batch, but for the data of the query set.
                This argument might be optional depending on how the learner is implemented.

        Returns: 
            * loss (torch.tensor): A tensor containing the loss that results from the inner loop 
        """
        raise NotImplementedError()

    ###### Model evaluation methods ######

    @abc.abstractmethod
    def run_finetuning_classification(self, finetune_dataloader, *args, **kwargs):
        """
        Finetunes the model on the data of finetune_dataloader.

        Args:
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model is
                passed in as a dataloader (in most cases this will be an NLUDataloader)

        Returns:
            * inference_params ({}): Returns any sort of params that need to be used for the 
                run_inference_classification method 
        """ 
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inference_classification(self, inference_dataloader, *args, **kwargs):
        """
        Evaluates the model on the data of inference_dataloader.

        Args: 
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)

        Returns: 
            * predictions ([int]): A dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): The value of the classification loss on the inference dataset.
        """
        raise NotImplementedError()

