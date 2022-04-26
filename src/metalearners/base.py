__author__ = 'Richard Diehl Martinez'
""" Base ABC Class for (meta) learners """

import abc 
import os
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ..taskheads import ClassificationHead

class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(self, base_model, base_device, **kwargs):
        """
        BaseLearner establishes the inferface for the learner class and 
        inherents from torch.nn.Module. 
        
        Args: 
            * base_model (implements a BaseModel)
            * base_device (str): what device to use for training (either 'cpu' or 'gpu) 
        """
        super().__init__()

        # hidden dimensions of the outputs of the base_model
        self.base_model = base_model
        self.base_model.to(base_device)
        self.base_model_hidden_dim = base_model.hidden_dim 

        self.base_device = base_device

    ###### Task head initialization methods ######

    def get_task_init_kwargs(self, n_classes, **kwargs):
        """ 
        Helper method for generating keyword arguments that can be passed into a task head 
        initialization method
        
        Returns:
            * init_kwargs (dict): Keyword arguments used by the task head initialization function 
        """

        # The two attributes below need to be specified by the learner
        assert(hasattr(self, 'base_model_hidden_dim') and hasattr(self, 'base_device'))

        init_kwargs = {}

        init_kwargs['base_model_hidden_dim'] = self.base_model_hidden_dim
        init_kwargs['n_classes'] = n_classes
        init_kwargs['device'] = self.base_device 

        return init_kwargs

    ###### Model training and evaluation helper methods ######

    @staticmethod
    def _compute_task_loss(model_outputs, data_batch, task_head_weights, task_type):
        """
        Helper function for computing the task loss on a given batch of data. We 
        assume that the data has already been passed through the base_model - the result of which
        is model_outputs (i.e. the final layer's hidden states). 

        Args: 
            * model_outputs (torch.Tensor): Result of passing data_batch through the 
                base_model. Should have shape: (batch_size, sequence_length, hidden_size)
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * task_head_weights (dict): weights used by the task head (in this the classifier head)
            * task_type (str): Type of task (e.g. 'classification')
        Returns: 
            * logits ([torch.Tensor]): Logits resulting from forward pass 
            * loss (int): Loss of data 
        """

        #indexing into sequence layer of model_outputs -> (batch_size, hidden_size) 
        batch_size = model_outputs.size(0)
        last_hidden_state = model_outputs[torch.arange(batch_size),
                                              data_batch['input_target_idx']]

        if task_type == 'classification':
            head = ClassificationHead()
        else: 
            logger.exception(f"Invalid task type: {task_type}")
            raise Exception(f"Invalid task type: {task_type}")

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
    
    def forward(self, learner, support_batch, query_batch, device):
        """ 
        NOTE: Only torch.DistributedDataParallel should indirectly call this - used as a wrapper to 
              run_inner_loop. Unless you know what you're doing, don't call this method.
        """
        return learner.run_inner_loop(support_batch, query_batch, device)

    def setup_DDP(self, rank, world_size):
        """ 
        Helper method for setting up distributed data parallel process group and returning 
        a wrapper DDP instance of the learner
        
        Args: 
            * rank (int): Rank of current GPU 
            * world_size (int): Number of GPUs should be the same as utils.num_gpus

        Returns:
            * device (int): Device to run model on
            * ddp (torch.DistributedDataParallel): Wrapped DDP learner
        """
        device = f"cuda:{rank}"
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '32432'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

        self.to(device)

        ddp = DDP(self, device_ids=[rank], find_unused_parameters=True)
        return (device, ddp)

    @abc.abstractmethod
    def run_inner_loop_mp(self, rank, world_size, data_queue, loss_queue, num_tasks_per_iteration):
        """
        Entry point for running inner loop using multiple processes. Sets up DDP init process
        group, wraps learner in DDP and calls forward/backward on the DDP-wrapped model.

        Args: 
            * rank (int): Rank of current GPU 
            * world_size (int): Number of GPUs should be the same as utils.num_gpus
            * data_queue (multiprocessing.Queue): Queue from which we read passed in data
            * loss_queue (multiprocessing.Queue): Queue to which we write loss values
            * step_optimizer (multiprocessing.Event): Event to signal workers to take an optimizer
                step
            * num_tasks_per_iteration (int): Number of tasks per iteration that the user specifies
                in the experiment config file
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inner_loop(self, support_batch, query_batch=None, device=None, **kwargs): 
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
            * device: Optional string to specify a device to override base_device

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

