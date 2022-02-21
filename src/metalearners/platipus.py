__author__ = 'Richard Diehl Martinez '
"""
Implementation of the platipus model, proposed by Finn et el. https://arxiv.org/pdf/1806.02817.pdf

Implementation is lightly adapted from the open-source repo: 
https://github.com/cnguyen10/few_shot_meta_learning/blob/2b075a5e5de4f81670ae8340e87acfc4d5e9bbc3/Platipus.py#L28
"""

import typing
import higher 
import logging
import itertools

import torch
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLearner
from ..utils.modeling import kl_divergence_gaussians

logger = logging.getLogger(__name__)

class Platipus(BaseLearner):
    def __init__(self, base_model,  optimizer_type='adam',
                                    meta_lr=0.01,
                                    inner_lr=0.01,
                                    kl_weight=0.5,
                                    loss_function="cross_entropy",
                                    num_inner_steps=3,
                                    use_first_order=False,
                                    task_embedding_method='val_grad',
                                    device="cpu",
                                    *args,
                                    **kwargs):
        """
        # TODO - add doc string 
        """
        super().__init__()

        self.device = device

        # getting functional form of the model
        self.functional_model = higher.patch.make_functional(module=base_model)

        # hidden dimensions of the outputs of the base_model
        self.base_model_hidden_dim = base_model.hidden_dim 
        
        # establishing meta parameters to-be learned
        self.mu_theta = [] 
        self.log_sigma_theta = []
        self.log_v_q = []
        for param in base_model.parameters():
            self.mu_theta.append(torch.nn.Parameter(data=torch.randn(size=param.shape).to(self.device), requires_grad=param.requires_grad))
            self.log_sigma_theta.append(torch.nn.Parameter(data=torch.randn(size=param.shape).to(self.device), requires_grad=param.requires_grad))
            self.log_v_q.append(torch.nn.Parameter(data=torch.randn(size=param.shape).to(self.device), requires_grad=param.requires_grad))

        self.gamma_p = torch.nn.Parameter(data=torch.tensor(0.01).to(self.device))
        self.gamma_q = torch.nn.Parameter(data=torch.tensor(0.01).to(self.device))

        self.all_meta_params = itertools.chain(self.mu_theta, self.log_sigma_theta, self.log_v_q, [self.gamma_p, self.gamma_q])

        # loading in meta optimizer 
        self.meta_lr = float(meta_lr)
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=self.all_meta_params, lr=self.meta_lr)
        else: 
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

        # fixed learning rate used for finetune adapting task-specific phi weights
        self.inner_lr = float(inner_lr)

        # hyper-param for trading off ce-loss and kl-loss
        self.kl_weight = float(kl_weight)

        # defining loss function 
        if loss_function == 'cross_entropy':
            self.loss_function = torch.nn.CrossEntropyLoss()
        else: 
            raise Exception(f"Invalid loss function: {loss_function}")

        # number of steps to perform in the inner loop
        self.num_inner_steps = int(num_inner_steps)
        
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
        # TODO: check if this is equivalent to original implementation
        cloned_params = [torch.clone(p) for p in params]

        for _ in range(self.num_inner_steps): 
            outputs = self.functional_model.forward(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    params=cloned_params)

            # last_hidden_state has form (batch_size, sequence_length, hidden_size);
            # where hidden_size = self.base_model_hidden_dim
            last_hidden_state = outputs.last_hidden_state

            # indexing into sequence layer of last_hidden state -> (batch_size, hidden_size) 
            batch_size = last_hidden_state.size(0)
            last_hidden_state = last_hidden_state[torch.arange(batch_size), input_target_idx]

            # (batch_size, num_classes) 
            logits = task_classifier(last_hidden_state)

            loss = self.loss_function(input=logits, target=label_ids)
            grad_params = [p for p in cloned_params if p.requires_grad]

            if self.use_first_order:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=grad_params,
                    retain_graph=True, # TODO: check if this is required
                    allow_unused=True,
                )
            else:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=grad_params,
                    create_graph=True,
                    allow_unused=True,
                )
            
            # updating params
            grad_counter = 0
            for idx, p in enumerate(cloned_params):
                if p.requires_grad:
                    cloned_params[idx] = cloned_params[idx] - learning_rate * grads[grad_counter]
                    grad_counter += 1
        
        return cloned_params

    def optimizer_step(self, set_zero_grad=True):
        """ Take a global update step of self.params """
        self.optimizer.step()
        if set_zero_grad:
            self.optimizer.zero_grad()

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
        task_classifier = torch.nn.Linear(self.base_model_hidden_dim, n_classes).to(self.device)

        # TODO: implement task embedding 
        # mu theta adapted to the query set
        mu_theta_query = self.adapt_params(**query_batch, params=self.mu_theta,
                                                          task_classifier=task_classifier,
                                                          learning_rate=self.gamma_q)
        
        # sampling from task specific distribution updated mu_theta 
        theta = [None] * len(self.mu_theta) 
        for i in range(len(theta)):
            theta[i] = mu_theta_query[i] + \
                torch.randn_like(input=mu_theta_query[i], device=self.device) * torch.exp(input=self.log_v_q[i])

        # TODO - adapt the task_classifier this time 
        # adapting to the support set 
        phi = self.adapt_params(**support_batch, params=theta,
                                                 task_classifier=task_classifier,
                                                 learning_rate=self.inner_lr)

        # computing CE loss on the query set with updated phi params
        outputs = self.functional_model.forward(input_ids=query_batch['input_ids'],
                                                attention_mask=query_batch['attention_mask'],
                                                params=phi)

        last_hidden_state = outputs.last_hidden_state
        batch_size = last_hidden_state.size(0)
        last_hidden_state = last_hidden_state[torch.arange(batch_size), query_batch['input_target_idx']]
        logits = task_classifier(last_hidden_state)

        ce_loss = self.loss_function(input=logits, target=query_batch['label_ids'])

        # computing KL loss (requires adapting mu_theta to the support set)
        
        # mu theta adapted to the support set (line 10 from algorithm 1)
        mu_theta_support = self.adapt_params(**support_batch, params=self.mu_theta,
                                                              task_classifier=task_classifier,
                                                              learning_rate=self.gamma_p)

        kl_loss = kl_divergence_gaussians(p=[*mu_theta_query, *self.log_v_q], q=[*mu_theta_support, *self.log_sigma_theta])

        loss = ce_loss + self.kl_weight * kl_loss
        return loss

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

    
