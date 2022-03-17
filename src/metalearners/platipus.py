__author__ = 'Richard Diehl Martinez'
"""
Implementation of the platipus model, proposed by Finn et el. https://arxiv.org/pdf/1806.02817.pdf

Implementation is adapted from the open-source repo: 
https://github.com/cnguyen10/few_shot_meta_learning/blob/2b075a5e5de4f81670ae8340e87acfc4d5e9bbc3/Platipus.py#L28
"""

import typing
import higher 
import logging
import itertools
import math

import torch
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter

from .base import BaseLearner
from ..utils.modeling import kl_divergence_gaussians
from ..utils import move_to_device

logger = logging.getLogger(__name__)

class Platipus(BaseLearner):
    def __init__(self, base_model,  optimizer_type='adam',
                                    meta_lr=0.01,
                                    inner_lr=0.01,
                                    kl_weight=0.5,
                                    num_inner_steps=3,
                                    use_first_order=False,
                                    task_embedding_method='val_grad',
                                    device="cpu",
                                    *args,
                                    **kwargs):
        """
        Platipus implements a type of BaseLearner.

        Reads in a base model and converts the model into a functional 'state-less' version of the model, using 
        the higher library. Platipus requires tracking meta parameters that represent the mean and 
        standard deviations of the weights that are to-be meta-learned. These meta parameters are stored 
        as internal attributes of platipus, and are learned via an optimizer of type (optimizer_type). The main
        logic of platipus is in the run_inner_loop() and run_evaluation() methods which implement the training 
        and inference steps of platipus. 

        Args: 
            * base_model (implementation of BaseModel)
            * optimizer_type (str) : the type of the optimizer (defaults to 'adam')
            * meta_lr (float): learning rate for the outer loop meta learning step (defaults to 0.01)
            * inner_lr (float): learning rate for the inner loop adaptation step (defaults to 0.01)
            * kl_weight (float): trade-off hyper-parameter between the CE term and the KL term of the ELBO (defaults to 0.5)
            * num_inner_steps (int): number of gradients steps in the inner looop
            * use_first_order (bool): whether a first order approximation of higher-order gradients should be used (defaults to False)
            * task_embedding_method (str): how to represent the task embedding that is used to condition the task-dependent 
                probability distribution from which we sample weights for a given task (defaults to 'val_grad'). 'Val_grad' is the 
                implicit method used in the platipus paper. 
            * device (str): what device to use for training (either 'cpu' or 'gpu) - defaults to 'cpu'

        """

        super().__init__()

        self.device = device

        # getting functional form of the model 
        self.functional_model = higher.patch.make_functional(module=base_model)

        # NOTE: these lines are super important (otherwise computation graph explodes)
        self.functional_model.track_higher_grads = False
        self.functional_model._fast_params = [[]]

        # hidden dimensions of the outputs of the base_model
        self.base_model_hidden_dim = base_model.hidden_dim 
        
        # establishing meta parameters to-be learned
        self.mu_theta = [] 
        self.log_sigma_theta = []
        self.log_v_q = []
        for param in base_model.parameters():
            self.mu_theta.append(param) # global mean parameters initialized as the weights of the base model
            self.log_sigma_theta.append(torch.nn.Parameter(data=torch.randn(size=param.shape).to(self.device) - 4, requires_grad=param.requires_grad))
            self.log_v_q.append(torch.nn.Parameter(data=torch.randn(size=param.shape).to(self.device) - 4, requires_grad=param.requires_grad))

        self.gamma_p = torch.nn.Parameter(data=torch.tensor(1e-2).to(self.device))
        self.gamma_q = torch.nn.Parameter(data=torch.tensor(1e-2).to(self.device))

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
        self.ce_loss_function = torch.nn.CrossEntropyLoss()

        # number of steps to perform in the inner loop
        self.num_inner_steps = int(num_inner_steps)
        
        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            use_first_order = eval(use_first_order)

        self.use_first_order = use_first_order

        self.task_embedding_method = task_embedding_method


    def _run_forward_pass(self, data_batch, params, task_classifier_weights):
        """ 
        Helper method for running a batch of data through a forward pass of the functional model 

        Args: 
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * params ([torch.Tensor]): List of tensor weights storing the model weights 
            * task_classifier_weights (dict): weights of classifier layer; see 
                _initialize_task_classifier_weights for explanation of dict values

        Returns:
            * logits ([torch.Tensor]): Logits resulting from forawrd pass 
            * loss (int): Loss of data 
        """

        outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                attention_mask=data_batch['attention_mask'],
                                                params=params)

        # last_hidden_state has form (batch_size, sequence_length, hidden_size);
        # where hidden_size = self.base_model_hidden_dim
        last_hidden_state = outputs.last_hidden_state

        # indexing into sequence layer of last_hidden state -> (batch_size, hidden_size) 
        batch_size = last_hidden_state.size(0)
        last_hidden_state = last_hidden_state[torch.arange(batch_size), data_batch['input_target_idx']]

        # (batch_size, num_classes) 
        logits = F.linear(last_hidden_state, **task_classifier_weights)
        loss = self.ce_loss_function(input=logits, target=data_batch['label_ids'])

        return (logits, loss)


    def _initialize_task_classifier_weights(self, n_classes, method='xavier_normal'):
        """ 
        Helper method for initializing the weight and bias layer of the final classification layer
        to classify n_classes number of classes for the current task.

        Args: 
            * n_classes (int): Number of classes 
            * method (str): Method for weight initialization
        Returns: 
            * task_classifier_weights (dict): {
                * weight -> (torch.Tensor): classification weight matrix
                * bias -> (torch.Tensor): classification bias vector
                }
        """

        # TODO allow task classifier to be initialized with some 'smart weights' (protoMAML style)

        if method == 'xavier_normal':
            # Xavier normal weight implementation
            std_weight = math.sqrt(2.0 / float(self.base_model_hidden_dim + n_classes))
            std_bias = math.sqrt(2.0 / float(n_classes))

            # weights need to be shape (out_features, in_features) to be compatible with functional linear
            classifier_weight = torch.randn((n_classes, self.base_model_hidden_dim), device=self.device) * std_weight
            classifier_bias = torch.randn((n_classes), device=self.device) * std_bias
        else:
            raise NotImplementedError(f"Task classification weight initialization method: {method} not supported")

        classifier_weight.requires_grad = True
        classifier_bias.requires_grad = True

        task_classifier_weights = { 
            "weight": classifier_weight,
            "bias": classifier_bias
        }

        return task_classifier_weights
        
    ###### Model adaptation methods ######

    def sample_adapted_weights(self, data_batch, sampling_std, return_adapted_mean=False, **adaptation_kwargs): 
        """ 
        Helper method for sampling model weights that have been adapted to a batch of 
        data. 

        Args: 
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * sampling_std ([torch.Tensor]): The standard deviation to be used for sampling weights
                for each weight that is to-be meta-learned
            * return_adapted_mean (bool): Whether to also return the means from the adapted distribution 
                over weights (defaults to False)
            * adaptation_kwargs: Kerword arguments to be passed into adapt_params_classification 
                (see this method for more information) 
        Returns: 
            * adapted_params ([torch.Tensor]): Adapted means of the weight distribution (only returned 
                if return_adapted_mean is set to True)
            * sampled_weights ([torch.Tensor]): The generated list of tensor weights sampled 
                from the model 
        """

        adapted_params = self.adapt_params_classification(data_batch, **adaptation_kwargs)

        sampled_weights = [None] * len(adapted_params) 

        for i in range(len(sampled_weights)):
            if adapted_params[i].requires_grad: 
                sampled_weights[i] = adapted_params[i] + \
                    torch.randn_like(input=adapted_params[i], device=self.device) * torch.exp(input=sampling_std[i])
            else: 
                sampled_weights[i] = self.mu_theta[i]

        if return_adapted_mean:
            return (adapted_params, sampled_weights)
        else: 
            return sampled_weights  

    # for adapting to classificationt tasks (e.g. the meta-training task is a classification task)
    def adapt_params_classification(self, data_batch, params, task_classifier_weights, learning_rate, 
                                                                                       optimize_classifier=False,
                                                                                       clone_params=True,
                                                                                       evaluation_mode=False,
                                                                                       override_num_inner_steps=0):
        """ 
        Adapted from: 
        https://github.com/cnguyen10/few_shot_meta_learning/blob/2b075a5e5de4f81670ae8340e87acfc4d5e9bbc3/Platipus.py#L28

        For a given batch of inputs and labels, we compute the loss of the functional model where the parameters of the 
        model are overriden to be params. Note that this is the main reason that in __init__ we convert the base model
        to a functional version of the model that can take in its parameters as an argument (as opposed to these 
        parameters being stored in the model's state). 

        The parameters are then updated using SGD with a given learning rate, and returned. 

        Params:
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * params (iterable): Iterable of torch.nn.Paramater()s (the params that we want to evaluate our model at)
            * task_classifier_weights (dict):  The final classification layer which is not meta-learned 
                See _initialize_task_classifier_weights for a discription of the dict structure
            * learning_rate (float): The internal learning rate used to update params
            * optimize_classifier (bool): Whether to update the the final classification layer - this layer is not meta-learned
                (defaults to False)
            * clone_params (bool): Whether to clone the params passed in (defaults to True)
            * evaluation_model(bool): Wheter running this method during evaluation (either in finetuning or inference) (defaults to False)
            * override_num_inner_steps (int): If set to a value above 0 will override self.num_inner_steps (defaults to 0)

        Returns: 
            * adapted_params (iterable): Iterable of torch.nn.Paramater()s that represent the updated the parameters 
                after running SGD 
        """

        if clone_params:
            # copy the parameters but allow gradients to propagate back to original params
            adapted_params = [torch.clone(p) for p in params]
        else:
            adapted_params = params 

        if override_num_inner_steps:
            num_inner_steps = override_num_inner_steps
        else: 
            num_inner_steps = self.num_inner_steps

        for _ in range(num_inner_steps):

            logits, loss = self._run_forward_pass(data_batch, params=adapted_params, 
                                                              task_classifier_weights=task_classifier_weights)

            grad_params = [p for p in adapted_params if p.requires_grad]

            if self.use_first_order or evaluation_mode:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=grad_params,
                    retain_graph=True,
                )
            else:
                grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=grad_params,
                    create_graph=True,
                )

            # updating params
            grad_counter = 0
            for i, p in enumerate(adapted_params):
                if p.requires_grad:
                    adapted_params[i] = adapted_params[i] - learning_rate * grads[grad_counter]
                    grad_counter += 1

            # optionally use SGD to update the weights of the final linear classification layer
            if optimize_classifier:
        
                classifier_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=[task_classifier_weights["weight"], task_classifier_weights["bias"]],
                )

                task_classifier_weights["weight"] = task_classifier_weights["weight"] - learning_rate * classifier_grads[0]
                task_classifier_weights["bias"] =  task_classifier_weights["bias"] - learning_rate * classifier_grads[1]

        return adapted_params


    ###### Model training methods ######

    def optimizer_step(self, set_zero_grad=True):
        """ Take a global update step of self.params; optionally set gradients back to 0"""
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
        task_classifier_weights = self._initialize_task_classifier_weights(n_classes)

        # TODO: implement task embedding 
        
        # sampling theta after adapting model to query_batch
        mu_theta_query, theta = self.sample_adapted_weights(query_batch, sampling_std=self.log_v_q,
                                                                         return_adapted_mean=True,
                                                                         params=self.mu_theta,
                                                                         task_classifier_weights=task_classifier_weights,
                                                                         learning_rate=self.gamma_q,
                                                                         clone_params=True)

        # adapting theta to the support set -> adapted params are phi
        phi = self.adapt_params_classification(support_batch, 
                                               params=theta, 
                                               task_classifier_weights=task_classifier_weights,
                                               learning_rate=self.inner_lr,
                                               clone_params=False,
                                               optimize_classifier=True)

        # evaluating on the query batch using the adapted params phi  
        logits, ce_loss = self._run_forward_pass(query_batch, params=phi, task_classifier_weights=task_classifier_weights)

        # computing KL loss (requires adapting mu_theta to the support set)
        
        # mu theta adapted to the support set (line 10 from algorithm 1) 
        mu_theta_support = self.adapt_params_classification(support_batch, 
                                                            params=self.mu_theta,
                                                            task_classifier_weights=task_classifier_weights,
                                                            learning_rate=self.gamma_p,
                                                            clone_params=True)

        kl_loss = kl_divergence_gaussians(p=[*mu_theta_query, *self.log_v_q], q=[*mu_theta_support, *self.log_sigma_theta])

        loss = ce_loss + self.kl_weight * kl_loss

        return loss


    ###### Model evaluation methods ######

    # for NLU classification tasks

    def run_finetuning_classification(self, finetune_dataloader, n_classes, **kwargs): 
        """
        Creates a copy of the trained model parameters and continues to finetune these 
        parameters on a given dataset. This method assumes that the task that is being 
        finetuned is a classification task (e.g. NLI). Effectively, implements a 
        slightly adapted version of algorithm 2 of the platipus paper
        https://arxiv.org/pdf/1806.02817.pdf for meta-testing.
        
        Args: 
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model is passed
                in as a dataloader (in most cases this will be an NLUDataloader)
            * n_classes (int): the number of classes to classify over

        Returns:
            * inference_params dict containing: 
                * finetuned_params ([nn.Parameter]): List of the finetuned model's parameters
                * task_classifier_weights (dict): weights of classifier layer; see 
                    _initialize_task_classifier_weights for explanation of dict values
        """
        task_classifier_weights = self._initialize_task_classifier_weights(n_classes)

        finetuned_theta = None
    
        for batch_idx, data_batch in enumerate(finetune_dataloader):
            data_batch = move_to_device(data_batch, self.device)

            if batch_idx == 0:
                # use the first batch of the finetuning dataset to sample weights 
                finetuned_theta = self.sample_adapted_weights(data_batch, sampling_std=self.log_sigma_theta,
                                                                          params=self.mu_theta,
                                                                          task_classifier_weights=task_classifier_weights,
                                                                          learning_rate=self.gamma_p,
                                                                          clone_params=True,
                                                                          evaluation_mode=True)
            
            # run SGD on the finetuned theta parameters
            finetuned_theta = self.adapt_params_classification(data_batch, 
                                                               params=finetuned_theta,
                                                               task_classifier_weights=task_classifier_weights,
                                                               learning_rate=self.inner_lr,
                                                               optimize_classifier=True,
                                                               clone_params=False,
                                                               evaluation_mode=True,
                                                               override_num_inner_steps=1)

            # TODO: remove me 
            if batch_idx == 2:
                break

        inference_params = {
            "finetuned_params": finetuned_theta,
            "task_classifier_weights": task_classifier_weights
        }

        return inference_params


    def run_inference_classification(self, inference_dataloader, finetuned_params=None, 
                                                                 task_classifier_weights=None,
                                                                 adaptation_batch=None, 
                                                                 **kwargs):
        """ 
        In most use-cases this method will be called after the run_finetuning_classification. 
        As the name suggests, this method runs inference on an NLU dataset for some classification
        task (like NLI). The primary adaptation of the weights for a given task should 
        occur in the run_finetuning_classification method. It is possible, however, to 
        adapt the weights one more time on a given dataset, by passing in a batch of data 
        (adaptation_batch). 

        Args: 
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)
            * finetuned_params ([nn.Parameter]): List of the finetuned model's parameters
            * task_classifier_weights (dict): weights of classifier layer; see 
                _initialize_task_classifier_weights for explanation of dict values
            * adaptation_batch: a dictionary containing the required data for running a forward 
                pass of the model (see run_inner_loop for explanation of the dictionary)

        Returns: 
            * predictions ([int]): a dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): the value of the classification loss on the inference dataset.
        """

        if finetuned_params is None: 
            finetuned_params = self.mu_theta

        if task_classifier_weights is None: 
            task_classifier_weights = self._initialize_task_classifier_weights(kwargs['n_classes'])

        if adaptation_batch is not None:
                # if adaptation_batch is passed in, we adapt the model's parameters to this data
                adaptation_batch = move_to_device(adaptation_batch, self.device)

                finetuned_params = self.sample_adapted_weights(adaptation_batch, sampling_std=self.log_sigma_theta,
                                                                                 params=finetuned_params,
                                                                                 task_classifier_weights=task_classifier_weights,
                                                                                 learning_rate=self.gamma_p,
                                                                                 clone_params=True,
                                                                                 evaluation_mode=True)
        
        predictions = []
        total_loss = 0.0
        total_samples = 0

        # Running final inference script over the evaluation data
        with torch.no_grad():
            for data_batch in inference_dataloader: 
                data_batch = move_to_device(data_batch, self.device)

                logits, loss = self._run_forward_pass(data_batch, params=finetuned_params, 
                                                                  task_classifier_weights=task_classifier_weights)

                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

        return (predictions, total_loss)


