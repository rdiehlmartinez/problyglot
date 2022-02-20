"""
PLATIPUS Model Implementation: https://arxiv.org/pdf/1806.02817.pdf
Copied from open-source repo: 
https://github.com/cnguyen10/few_shot_meta_learning/blob/2b075a5e5de4f81670ae8340e87acfc4d5e9bbc3/Platipus.py#L28
"""

import typing
import higher 
import logging
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLearner
from ..utils import device

logger = logging.getLogger(__name__)

class Platipus(BaseLearner):
    def __init__(self, base_model,  optimizer_type='adam',
                                    meta_lr=0.01,
                                    loss_function="cross_entropy",
                                    num_inner_steps=3,
                                    use_first_order=False,
                                    task_embedding_method='val_grad',
                                    *args,
                                    **kwargs):
        """
        # TODO - add doc string 
        """

        # getting functional form of the model
        self.functional_model = higher.patch.make_functional(module=base_model)

        # hidden dimensions of the outputs of the base_model
        self.base_model_hidden_dim = base_model.hidden_dim 

        # establishing meta parameters to-be learned
        self.mu_theta = [] 
        self.log_sigma_theta = []
        self.log_v_q = []
        for param in base_model.parameters():
            self.mu_theta.append(torch.nn.Parameter(data=torch.randn(size=param.shape), requires_grad=param.requires_grad).to(device))
            self.log_sigma_theta.append(torch.nn.Parameter(data=torch.randn(size=param.shape), requires_grad=param.requires_grad).to(device))
            self.log_v_q.append(torch.nn.Parameter(data=torch.randn(size=param.shape), requires_grad=param.requires_grad).to(device))

        self.gamma_p = [torch.nn.Parameter(data=torch.tensor(0.01)).to(device)]
        self.gamma_q = [torch.nn.Parameter(data=torch.tensor(0.01)).to(device)]

        all_params = itertools.chain(self.mu_theta, self.log_sigma_theta, self.log_v_q, self.gamma_p, self.gamma_q)

        # loading in meta optimizer 
        self.meta_lr = float(meta_lr)
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=all_params, lr=self.meta_lr)
        else: 
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

        self.num_inner_steps = int(num_inner_steps)
        
        # defining loss function 
        if loss_function == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else: 
            raise Exception(f"Invalid loss function: {loss_function}")
        
        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            use_first_order = eval(use_first_order)

        self.use_first_order = use_first_order

        self.task_embedding_method = task_embedding_method
        
        
    def adapt_params(self, input_ids, input_target_idx, attention_mask, label_ids,
                           params, task_classifier, learning_rate):
        """ 
        Adapted from: 
        https://github.com/cnguyen10/few_shot_meta_learning/blob/2b075a5e5de4f81670ae8340e87acfc4d5e9bbc3/Platipus.py#L28
        """

        # copy the parameters but allow gradients to propagate back to original params
        cloned_params = [torch.clone(p) for p in params]

        for _ in range(self.num_inner_steps): 
            print("inside adapt params - passing inputs to model")
            outputs = self.functional_model.forward(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    params=cloned_params)

            # last_hidden_state has form (batch_size, sequence_length, hidden_size);
            # note hidden_size = self.base_model_hidden_dim
            last_hidden_state = outputs.last_hidden_state
            logits = task_classifier(last_hidden_state)

            print(logits.shape)
            print(input_target_idx)
            print(input_ids[0])
            print(input_ids[0][input_target_idx[0]])
            exit()

            loss = self.loss_function(input=logits, target=labels)

            if self.use_first_order:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=params_copy,
                    retain_graph=True
                )
            else:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=params_copy,
                    create_graph=True
                )
            
            # take inner-loop update step
            for i in range(len(params_copy)):
                cloned_params[i] = cloned_params[i] - learning_rate * grads[i]
        
        return cloned_params

    def optimizer_step(self, set_zero_grad=True):
        """ Take a global update step of self.params """
        self.optimizer.step()
        if set_zero_grad:
            self.optimzer.zero_grad()

    def run_inner_loop(self, support_batch, query_batch, *args, **kwargs): 
        """
        Implements steps 6-11 outlined in the algorithm 1 of the Platipus paper https://arxiv.org/pdf/1806.02817.pdf, 
        i.e. the inner loop optimization step of the platipus algorithm.
        
        Args: 
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index we apply 
                    the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
            * query_batch: same as support_batch, but for the data of the query set 

        Returns: 
            * loss (torch.Tensor): loss value of the inner loop calculations
        """

        # automatically infer the number N of classes
        n_classes = torch.unique(support_batch['label_ids']).numel()

        # TODO allow task classifier to be initialized with some 'smart weights' (protoMAML style)
        task_classifier = torch.nn.Linear(self.base_model_hidden_dim, n_classes)

        # TODO: implement task embedding 
        update_mu_theta = self.adapt_params(**support_batch, params=self.mu_theta,
                                                             task_classifier=task_classifier,
                                                             learning_rate=self.gamma_q)
        

        # sampling using updated mu_theta 
        # TODO
        # theta = [None] * len(self.params[0]) 
        # for i in range(len(theta)):
        #     theta[i] = updated_mu_theta[i] + \
        #         torch.randn_like(input=updated_mu_theta[i], device=device) * torch.exp(input=self.params[2]) 
        
        # phi = self.adapt_params(support_inputs, support_labels, params=self.params[0], learning_rate=self.params[3])

    def run_evaluation(self, support_batch, query_batch, *args, **learner_kwargs):
        """ 
        Implements algorithm 2 of the platipus paper https://arxiv.org/pdf/1806.02817.pdf for meta-testing
                
        Args: 
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index we apply 
                    the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
            * query_batch: same as support_batch, but for the data of the query set 
        
        Returns: 
            * #TODO
        """
        # TODO 
        pass

    
