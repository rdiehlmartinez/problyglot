__author__ = 'Richard Diehl Martinez'
""" Implements a standard fully-supervised learning process (i.e. a baseline)"""

import copy
import torch
import itertools

from .base import BaseLearner
from ..taskheads import TaskHead
from ..utils import move_to_device

class BaselineLearner(BaseLearner):

    def __init__(self, base_model, optimizer_type='adam',
                                   lr=1e-2,
                                   *args,
                                   **kwargs): 
        """
        BaselineLearner implements a fully-supervised learning process to train
        the base_model (serves as a baseline).

        Args: 
            * base_model (implementation of BaseModel)
            * optimizer_type (str): The type of optimizer (e.g. 'adam')
            * lr (int): Learning rate of the optimizer
        """
        
        super().__init__(base_model, *args, **kwargs)

        self.base_model = base_model 

        # setting up optimizer
        params = [p for p in base_model.parameters() if p.requires_grad]
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=params, lr=float(lr))
        else: 
            logger.exception(f"Invalid optimizer type: {optimizer_type}")
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

    ###### Model training methods ######
 
    def run_inner_loop(self, support_batch, query_batch=None, *args, **kwargs): 
        """ 
        Run an inner loop optimization step. Usually this is in the context of 
        meta-learning, but in the case of a baseline model an inner_loop simply amounts 
        to a running a forward pass through the model and returning the corresponding loss.

        Args:
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch [optional]: same as support_batch, but for the data of the query set.
                This argument os optional in the case of the baseline model. If it is provided, we 
                simply concatenate this data and the support batch data together.

        Returns: 
            * loss (torch.tensor): a tensor containing the loss that results from the inner loop 
        """

        self.base_model.train()

        # combine support and query batch together 
        if query_batch: 
            input_batch = {}

            for key in support_batch.keys():
                support_batch_tensor = support_batch[key]
                query_batch_tensor = query_batch[key]

                if key == "input_ids" or key == "attention_mask":
                    # For input_ids and attention_mask we need to make sure that 
                    # the first dimension (sequence length dim) is the same 
                    # so we pad the shorter dim  
                    batch_size = support_batch_tensor.size(0)
                    max_seq_len_support = support_batch_tensor.size(1)
                    max_seq_len_query = query_batch_tensor.size(1)

                    if max_seq_len_support != max_seq_len_query: 
                        tensor_dim_diff = abs(max_seq_len_support - max_seq_len_query)
                        expansion_tensor_dims = (batch_size, tensor_dim_diff)
                        if key == "input_ids": 
                            expansion_tensor = torch.ones(expansion_tensor_dims,
                                                          device=self.device)
                        else: 
                            expansion_tensor = torch.zeros(expansion_tensor_dims,
                                                           device=self.device)

                        if max_seq_len_support > max_seq_len_query: 
                            # expanding query 
                            query_batch_tensor = torch.cat((query_batch_tensor, expansion_tensor),
                                                          dim=1).long()
                        else: 
                            # expanding support 
                            support_batch_tensor = torch.cat((support_batch_tensor,
                                                              expansion_tensor),
                                                              dim=1).long()
                    
                input_batch[key] = torch.cat((support_batch_tensor, query_batch_tensor), dim=0)
        
        else: 
            input_batch = support_batch

        n_classes = torch.unique(support_batch['label_ids']).numel()
        init_kwargs = self.get_task_init_kwargs(n_classes=n_classes)
        task_head_weights = TaskHead.initialize_task_head(task_type='classification',
                                                          method='random',
                                                          init_kwargs=init_kwargs)


        outputs = self.base_model(input_ids=data_batch['input_ids'],
                                  attention_mask=data_batch['attention_mask'])

        _, loss = self._compute_task_loss(outputs, data_batch, task_head_weights,
                                          task_type='classification')

        return loss

    ###### Model evaluation methods ######

    def run_finetuning_classification(self, finetune_dataloader, n_classes,
                                      max_finetuning_batch_steps=-1, **kwargs):
        """
        Finetunes the model on the data of finetune_dataloader.

        Args: 
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model
                is passed in as a dataloader (in most cases this will be an NLUDataloader)
            * n_classes (int): The number of classes to classify over
            * max_finetuning_batch_steps (int): Optional maximum number of batch steps to take 
                for model finetuning 

        Returns:
            * inference_params dict containing: 
                * finetuned_model ([torch.nn.Module]): Finetuned model
                * task_head_weights (dict): weights of classifier layer
        """ 
        self.base_model.train()

        init_kwargs = self.get_task_init_kwargs(n_classes=n_classes)
       
        task_head_weights = TaskHead.initialize_task_head(task_type='classification',
                                                          method='random',
                                                          init_kwargs=init_kwargs)

        finetuned_model = copy.deepcopy(self.base_model)

        finetuned_task_head_weights = {}
        for k, p in task_head_weights.items():
            detached_p = p.detach()
            detached_p.requires_grad = True
            finetuned_task_head_weights[k] = detached_p

        finetuned_model_params = [p for p in finetuned_model.parameters() if p.requires_grad]
        finetune_params = itertools.chain(finetuned_model_params,
                                          finetuned_task_head_weights.values())
        finetune_optimizer = torch.optim.Adam(params=finetune_params)

        for batch_idx, data_batch in enumerate(finetune_dataloader):
            data_batch = move_to_device(data_batch, self.device)
            finetune_optimizer.zero_grad()

            # run SGD on the finetuned theta parameters
            outputs = finetuned_model(input_ids=data_batch['input_ids'],
                                      attention_mask=data_batch['attention_mask'],)

            _, loss = self._compute_task_loss(outputs, data_batch, finetuned_task_head_weights,
                                              task_type='classification')

            loss.backward()
            finetune_optimizer.step()

            if max_finetuning_batch_steps > 0 and (batch_idx + 1) >= max_finetuning_batch_steps:
                break

        inference_params = {
            "finetuned_model": finetuned_model, 
            "task_head_weights": finetuned_task_head_weights
        }

        return inference_params

    def run_inference_classification(self, inference_dataloader, finetuned_model,
                                     task_head_weights, **kwargs):
        """
        This method is to be called after the run_finetuning_classification. Runs inference 
        on the data stored in inference_dataloader, using the finetuned_model.

        Args: 
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)
            * finetuned_model ([torch.nn.Module]): Finetuned model
            * task_head_weights (dict): weights of task head (classifier layer)

        Returns: 
            * predictions ([int]): a dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): the value of the classification loss on the inference dataset.
        """

        predictions = []
        total_loss = 0.0
        total_samples = 0

        # Running final inference script over the evaluation data
        with torch.no_grad():

            finetuned_model.eval()

            for data_batch in inference_dataloader: 
                data_batch = move_to_device(data_batch, self.device)

                outputs = finetuned_model(input_ids=data_batch['input_ids'],
                                          attention_mask=data_batch['attention_mask'],)

                logits, loss = self._compute_task_loss(outputs, data_batch, task_head_weights,
                                                       task_type='classification')
            
                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

            # reference to finetuned_model deleted - no need to set in train mode

        return (predictions, total_loss)