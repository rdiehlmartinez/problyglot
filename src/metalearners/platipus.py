__author__ = 'Richard Diehl Martinez'
"""
Implementation of the platipus model, proposed by Finn et el. https://arxiv.org/pdf/1806.02817.pdf

Implementation is adapted from the open-source project: 
https://github.com/cnguyen10/few_shot_meta_learning
"""

import typing
import higher 
import logging
import itertools
from collections import OrderedDict

import torch
import torch.nn.functional as F 

from .base import BaseLearner
from ..utils.modeling import kl_divergence_gaussians
from ..utils import move_to_device

logger = logging.getLogger(__name__)

class Platipus(BaseLearner):
    def __init__(self, base_model,  optimizer_type='adam',
                                    meta_lr=1e-2,
                                    gamma_p=1e-2,
                                    gamma_q=1e-2,
                                    inner_lr=1e-2,
                                    classifier_lr=1e-1,
                                    kl_weight=0.5,
                                    num_inner_steps=5,
                                    use_first_order=False,
                                    task_embedding_method='val_grad',
                                    task_cls_init_method='random',
                                    device="cpu",
                                    *args,
                                    **kwargs):
        """
        Platipus implements a type of BaseLearner.

        Reads in a base model and converts the model into a functional 'state-less' version
        of the model, using  the higher library. Platipus requires tracking meta parameters that
        represent the mean and standard deviations of the weights that are to-be meta-learned.
        These meta parameters are stored  as internal attributes of platipus, and are learned via
        an optimizer of type (optimizer_type). The main logic of platipus is in the
        run_inner_loop() and finetuning/evaluation methods which implement the training 
        and inference steps of platipus. 

        Args: 
            * base_model (implementation of BaseModel)
            * optimizer_type (str) : the type of the optimizer 
            * meta_lr (float): learning rate for the outer loop meta learning step
            * kl_weight (float): trade-off hyper-parameter between the CE term and the KL term of
                the ELBO objective 
            * num_inner_steps (int): number of gradients steps in the inner looop
            * use_first_order (bool): whether a first order approximation of higher-order gradients
                should be used (defaults to False)
            * task_embedding_method (str): how to represent the task embedding that is used to
                condition the task-dependent probability distribution from which we sample weights
                for a given task (defaults to 'val_grad'). 'Val_grad' is the implicit method used
                in the platipus paper. 
            * task_cls_init_method (str): how to initialize the final task classifier layer
                (either 'random' or 'protomaml'). 
            * device (str): what device to use for training (either 'cpu' or 'gpu) 

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
        self.mu_theta = torch.nn.ParameterList()
        self.log_sigma_theta = torch.nn.ParameterList()
        self.log_v_q = torch.nn.ParameterList()
        for param in base_model.parameters():
            self.mu_theta.append(param) # mean parameters initialized to weights of base model
            self.log_sigma_theta.append(torch.nn.Parameter(
                                            data=torch.randn(size=param.shape).to(self.device) - 4,
                                            requires_grad=param.requires_grad)
                                        )
            self.log_v_q.append(torch.nn.Parameter(
                                    data=torch.randn(size=param.shape).to(self.device) - 4,
                                    requires_grad=param.requires_grad)
                                )

        self.gamma_p = torch.nn.Parameter(data=torch.tensor(float(gamma_p)).to(self.device))
        self.gamma_q = torch.nn.Parameter(data=torch.tensor(float(gamma_q)).to(self.device))

        self.inner_lr = torch.nn.Parameter(data=torch.tensor(float(inner_lr)).to(self.device))
        self.classifier_lr = torch.nn.Parameter(data=torch.tensor(float(classifier_lr))\
                                .to(self.device))

        self.all_meta_params = itertools.chain(self.mu_theta, self.log_sigma_theta, self.log_v_q,
                                               [self.gamma_p, self.gamma_q,
                                                self.inner_lr, self.classifier_lr]
                                              )

        # loading in meta optimizer 
        self.meta_lr = float(meta_lr)
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=self.all_meta_params, lr=self.meta_lr)
        else: 
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

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

        self.task_cls_init_method = task_cls_init_method

    ###### Helper functions ######

    # Overriding nn.Module functionality 
    def state_dict(self):
        """ Overriding method to remove placeholder parameters from functional model"""
        original_state_dict = super().state_dict()
        updated_state_dict = OrderedDict()

        for key, val in original_state_dict.items():
            if "functional" in key:
                continue
            
            updated_state_dict[key] = val
        
        return updated_state_dict


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
            * logits ([torch.Tensor]): Logits resulting from forward pass 
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
        last_hidden_state = last_hidden_state[torch.arange(batch_size),
                                              data_batch['input_target_idx']]

        # (batch_size, num_classes) 
        logits = F.linear(last_hidden_state, **task_classifier_weights)
        loss = self.ce_loss_function(input=logits, target=data_batch['label_ids'])

        return (logits, loss)

    def get_task_init_kwargs(self, data_batch=None, **kwargs):
        """ 
        Overrides the parent's method to also include kwargs for protomaml

        Returns:
            * init_kwargs (dict): Keyword arguments used by the initialization function 
        """
        init_kwargs = super().get_task_init_kwargs(**kwargs)
        
        if self.task_cls_init_method == 'protomaml':
            assert(data_batch is not None),\
                "Use of protomaml as a classification head initializer requires a data_batch"

            init_kwargs['functional_model'] = self.functional_model
            init_kwargs['params'] = self.mu_theta
            init_kwargs['data_batch'] = data_batch

        return init_kwargs
        
    ###### Model adaptation methods ######

    """
    Platipus relies on these methods to adapt and sample from the weight distributions 
    that it meta-learned. 
    """

    def _sample_adapted_weights(self, data_batch, sampling_std, return_adapted_mean=False,
                                **adaptation_kwargs): 
        """ 
        Helper method for sampling model weights that have been adapted to a batch of 
        data. 

        Args: 
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * sampling_std ([torch.Tensor]): The standard deviation to be used for sampling weights
                for each weight that is to-be meta-learned
            * return_adapted_mean (bool): Whether to also return the means from the adapted
                distribution over weights (defaults to False)
            * adaptation_kwargs: Kerword arguments to be passed into adapt_params_classification 
                (see this method for more information) 
        Returns: 
            * adapted_params ([torch.Tensor]): Adapted means of the weight distribution
                (only returned if return_adapted_mean is set to True)
            * sampled_weights ([torch.Tensor]): The generated list of tensor weights sampled 
                from the model 
        """

        adapted_params = self._adapt_params_classification(data_batch, **adaptation_kwargs)

        sampled_weights = [None] * len(adapted_params) 

        for i in range(len(sampled_weights)):
            if adapted_params[i].requires_grad: 
                sampled_weights[i] = adapted_params[i] + \
                    torch.randn_like(input=adapted_params[i], device=self.device) *\
                        torch.exp(input=sampling_std[i])
            else: 
                sampled_weights[i] = self.mu_theta[i]

        if return_adapted_mean:
            return (adapted_params, sampled_weights)
        else: 
            return sampled_weights  

    # adaptation to classification tasks (e.g. the meta-training task is a classification task)
    def _adapt_params_classification(self, data_batch, params, task_classifier_weights,
                                     learning_rate, optimize_classifier=False, clone_params=True,
                                     evaluation_mode=False):
        """ 
        Adapted from: 
        https://github.com/cnguyen10/few_shot_meta_learning

        For a given batch of inputs and labels, we compute the loss of the functional model where
        the parameters of the model are overriden to be params. Note that this is the main reason
        that in __init__ we convert the base model to a functional version of the model that can
        take in its parameters as an argument (as opposed to these  parameters being stored in the
        model's state). 

        The parameters are then updated using SGD with a given learning rate, and returned. 

        Params:
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)
            * params (iterable): Iterable of torch.nn.Paramater()s
                (the params that we want to evaluate our model at)
            * task_classifier_weights (dict): The final classification layer that is not
                meta-learned 
            * learning_rate (float): The internal learning rate used to update params
            * optimize_classifier (bool): Whether to train the final classification layer. 
            * clone_params (bool): Whether to clone the params passed in (defaults to True)
            * evaluation_mode (bool): Whether running this method during evaluation
                (either in finetuning or inference) (defaults to False)

        Returns: 
            * adapted_params (iterable): Iterable of torch.nn.Paramater()s that represent the
                updated the parameters after running SGD 
        """                                        

        if clone_params:
            # copy the parameters but allow gradients to propagate back to original params
            adapted_params = [torch.clone(p) for p in params]
        else:
            adapted_params = params 

        for _ in range(self.num_inner_steps):

            _, loss = self._run_forward_pass(data_batch,
                                             params=adapted_params, 
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

                task_classifier_weights["weight"] = task_classifier_weights["weight"] -\
                                                        self.classifier_lr * classifier_grads[0]
                task_classifier_weights["bias"] =  task_classifier_weights["bias"] -\
                                                        self.classifier_lr * classifier_grads[1]

        return adapted_params


    ###### Model training methods ######

    def run_inner_loop(self, support_batch, query_batch, *args, **kwargs): 
        """
        Implements steps 6-11 outlined in the algorithm 1 of the Platipus paper
        https://arxiv.org/pdf/1806.02817.pdf, i.e. the inner loop optimization step.
        
        Args: 
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch: same as support_batch, but for the data of the query set 

        Returns: 
            * loss (torch.Tensor): loss value of the inner loop calculations
        """
        # automatically infer the number N of classes
        n_classes = torch.unique(support_batch['label_ids']).numel()

        task_init_data_batch = support_batch if self.task_cls_init_method == 'protomaml' else None
        init_kwargs = self.get_task_init_kwargs(n_classes=n_classes,
                                                data_batch=task_init_data_batch)
       
        task_classifier_weights = self.initialize_task_head(task_type='classification',
                                                            method=self.task_cls_init_method,
                                                            init_kwargs=init_kwargs)

        # sampling theta after adapting model to query_batch
        mu_theta_query, theta = self._sample_adapted_weights(query_batch, 
                                                             sampling_std=self.log_v_q,
                                                             return_adapted_mean=True,
                                                             params=self.mu_theta,
                                                             task_classifier_weights=\
                                                                task_classifier_weights,
                                                             learning_rate=self.gamma_q,
                                                             clone_params=True)

        # adapting theta to the support set -> adapted params are phi
        phi = self._adapt_params_classification(support_batch, 
                                                params=theta, 
                                                task_classifier_weights=task_classifier_weights,
                                                learning_rate=self.inner_lr,
                                                clone_params=False,
                                                optimize_classifier=True)

        # evaluating on the query batch using the adapted params phi  
        _, ce_loss = self._run_forward_pass(query_batch,
                                            params=phi,
                                            task_classifier_weights=task_classifier_weights)

        # computing KL loss (requires adapting mu_theta to the support set)
        
        # mu theta adapted to the support set (line 10 from algorithm 1) 
        mu_theta_support = self._adapt_params_classification(support_batch, 
                                                             params=self.mu_theta,
                                                             task_classifier_weights=\
                                                                task_classifier_weights,
                                                             learning_rate=self.gamma_p,
                                                             clone_params=True)

        kl_loss = kl_divergence_gaussians(p=[*mu_theta_query, *self.log_v_q],
                                          q=[*mu_theta_support, *self.log_sigma_theta])

        loss = ce_loss + self.kl_weight * kl_loss

        return loss


    ###### Model evaluation methods ######

    # for NLU classification tasks

    def run_finetuning_classification(self, finetune_dataloader, n_classes,
                                      max_finetuning_batch_steps=-1, **kwargs): 
        """
        Creates a copy of the trained model parameters and continues to finetune these 
        parameters on a given dataset. This method assumes that the task that is being 
        finetuned is a classification task (e.g. NLI). Effectively, implements a 
        slightly adapted version of algorithm 2 of the platipus paper
        https://arxiv.org/pdf/1806.02817.pdf for meta-testing.
        
        Args: 
            * finetune_dataloader (torch.data.Dataloader): The dataset for finetuning the model
                is passed in as a dataloader (in most cases this will be an NLUDataloader)
            * n_classes (int): The number of classes to classify over
            * max_finetuning_batch_steps (int): Optional maximum number of batch steps to take 
                for model finetuning 


        Returns:
            * inference_params dict containing: 
                * finetuned_params ([nn.Parameter]): List of the finetuned model's parameters
                * task_classifier_weights (dict): weights of classifier layer; see 
                    _initialize_task_classifier_weights for explanation of dict values
        """

        # task classifier weights are initialized randomly
        adaptation_batch = move_to_device(next(iter(finetune_dataloader)), self.device)

        task_init_data_batch = support_batch if self.task_cls_init_method == 'protomaml' else None
        init_kwargs = self.get_task_init_kwargs(data_batch=task_init_data_batch)
       
        task_classifier_weights = self.initialize_task_head(task_type='classification',
                                                            method=self.task_cls_init_method,
                                                            init_kwargs=init_kwargs)

        # sampling weights from mu_theta using the first batch of data 
        sampled_theta = self._sample_adapted_weights(adaptation_batch,
                                                     sampling_std=self.log_sigma_theta,
                                                     params=self.mu_theta,
                                                     task_classifier_weights=\
                                                        task_classifier_weights,
                                                     learning_rate=self.gamma_p,
                                                     clone_params=True,
                                                     evaluation_mode=True)

        # detaching parameters from original computation graph to create new leaf variables
        finetuned_theta = []
        for p in sampled_theta:
            detached_p = p.detach()
            detached_p.requires_grad = p.requires_grad
            finetuned_theta.append(detached_p)

        finetuned_task_classifier_weights = {}
        for k, p in task_classifier_weights.items():
            detached_p = p.detach()
            detached_p.requires_grad = True
            finetuned_task_classifier_weights[k] = detached_p

        finetune_params = itertools.chain(finetuned_theta,
                                          finetuned_task_classifier_weights.values())
        finetune_optimizer = torch.optim.Adam(params=finetune_params)

        for batch_idx, data_batch in enumerate(finetune_dataloader):
            data_batch = move_to_device(data_batch, self.device)
            finetune_optimizer.zero_grad()

            # run SGD on the finetuned theta parameters
            _, ce_loss = self._run_forward_pass(data_batch, 
                                                params=finetuned_theta,
                                                task_classifier_weights=\
                                                    finetuned_task_classifier_weights)

            ce_loss.backward()
            finetune_optimizer.step()

            if max_finetuning_batch_steps > 0 and (batch_idx + 1) >= max_finetuning_batch_steps:
                break

        inference_params = {
            "finetuned_params": finetuned_theta, 
            "task_classifier_weights": finetuned_task_classifier_weights
        }

        return inference_params


    def run_inference_classification(self, inference_dataloader, finetuned_params, 
                                     task_classifier_weights, adaptation_batch=None, **kwargs):
        """ 
        This method is to be called after the run_finetuning_classification. 
        
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
            * adaptation_batch: (optional) a dictionary containing the required data for running
                a forward pass of the model (see run_inner_loop for explanation of the dictionary)

        Returns: 
            * predictions ([int]): a dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): the value of the classification loss on the inference dataset.
        """

        if adaptation_batch is not None:
                # if adaptation_batch is passed in, we adapt the model's parameters to this data
                adaptation_batch = move_to_device(adaptation_batch, self.device)

                finetuned_params = self._sample_adapted_weights(adaptation_batch, 
                                                                sampling_std=self.log_sigma_theta,
                                                                params=finetuned_params,
                                                                task_classifier_weights=\
                                                                    task_classifier_weights,
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

                logits, loss = self._run_forward_pass(data_batch, 
                                                      params=finetuned_params, 
                                                      task_classifier_weights=\
                                                        task_classifier_weights)

                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

        return (predictions, total_loss)


# Registering new task head initialization method 
@BaseLearner.register_initialization_method
def classification_protomaml(base_model_hidden_dim, n_classes, functional_model, params,
                                data_batch, device, **kwargs):
    """
    Args: 
        * base_model_hidden_dim (int): The hidden dimensions of the outputs of the base_model 
        * n_classes (int): Number of classes to classify over 
        * functional_model (higher.MonkeyPatched): The 'functionalized' version of the 
            base model
        * params ([torch.Tensor]): List of tensor weights storing the model weights.
        * data_batch (dict): Batch of data for a forward pass through the model 
            (see run_inner_loop for information on the data structure).
        * device (str): Device type ('cuda' or 'cpu')
    Returns: 
        * task_classifier_weights (dict): {
            * weight -> (torch.Tensor): classification weight matrix
            * bias -> (torch.Tensor): classification bias vector
            }
    """

    outputs = functional_model.forward(input_ids=data_batch['input_ids'],
                                        attention_mask=data_batch['attention_mask'],
                                        params=[p for p in params])

    # last_hidden_state has form (batch_size, sequence_length, hidden_size);
    last_hidden_state = outputs.last_hidden_state
    batch_size = last_hidden_state.size(0)
    last_hidden_state = last_hidden_state[torch.arange(batch_size),
                                            data_batch['input_target_idx']]

    prototypes = torch.zeros((n_classes, base_model_hidden_dim), device=device)

    for c in range(n_classes):
        idx = torch.nonzero(data_batch['label_ids'] == c).squeeze()
        if idx.nelement() != 0:
            prototypes[c] = torch.mean(last_hidden_state[idx], dim=0)
        else:
            logger.warning("ProtoMaml weight initialization missing at least one class")

    classifier_weight = 2 * prototypes
    classifier_bias = -torch.norm(prototypes, dim=1)**2

    task_classifier_weights = { 
        "weight": classifier_weight,
        "bias": classifier_bias
    }

    return task_classifier_weights