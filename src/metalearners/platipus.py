__author__ = 'Richard Diehl Martinez'
"""
Implementation of the platipus model, proposed by Finn et el. https://arxiv.org/pdf/1806.02817.pdf

Code adapted from: https://github.com/cnguyen10/few_shot_meta_learning
"""

import time
import higher 
import logging
import itertools

from collections import OrderedDict
from multiprocessing.queues import Empty as EmptyQueue

import torch
import torch.distributed as dist

from .base import BaseLearner
from ..taskheads import TaskHead
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
                                    language_embedding_method='val_grad',
                                    language_head_init_method='random',
                                    lm_head_n=5,
                                    *args,
                                    **kwargs):
        """
        Platipus implements a type of BaseLearner.

        Reads in a base model and sets up all of the metalearners of the model that are going to 
        be learned, using the higher library. Platipus requires tracking meta parameters that
        represent the mean and standard deviations of the weights that are to-be meta-learned.
        These meta parameters are stored  as internal attributes of platipus, and are learned via
        an optimizer of type (optimizer_type). The main logic of platipus is in the
        run_inner_loop() and finetuning/evaluation methods which implement the training 
        and inference steps of platipus. 

        Args: 
            * base_model (implementation of BaseModel)
            * optimizer_type (str) : The type of optimizer (e.g. 'adam') 
            * meta_lr (float): Learning rate for the outer loop meta learning step
            * The following four keyword args define inner-loop hyper-parameters that can be 
                meta-learned. The values passed in represent the initial values:
                * gamma_p (float): std. dev of the gaussian dist. from which we sample weights 
                    conditioned on the support set 
                * gamma_q (float): std. dev of the gaussian dist. from which we sample weights 
                    conditioned on the query set 
                * inner_lr (float): inner-loop learning rate of the base_model 
                * classifier_lr (float): inner-loop learning rate of the classifier head
            * kl_weight (float): Trade-off hyper-parameter between the CE term and the KL term of
                the ELBO objective 
            * num_inner_steps (int): Number of gradients steps in the inner looop
            * use_first_order (bool): Whether a first order approximation of higher-order gradients
                should be used (defaults to False)
            * language_embedding_method (str): How to represent the embedding that is used to
                condition the language-dependent probability distribution from which we sample
                weights for a given language (defaults to 'val_grad'). 'Val_grad' is the implicit 
                method used in the platipus paper. 
            * language_head_init_method (str): How to initialize the language head classifier layer
            * lm_head_n (int): Size of n-way classification used for generating the language 
                modeling tasks used for training and conditioning the platipus model on a 
                given language  

        """

        super().__init__(base_model, *args, **kwargs)
        
        # establishing meta parameters to-be learned
        self.mu_theta = torch.nn.ParameterList()
        self.log_sigma_theta = torch.nn.ParameterList()
        self.log_v_q = torch.nn.ParameterList()

        for param in base_model.parameters():
            # mean parameters initialized to weights of base model
            self.mu_theta.append(torch.nn.Parameter(data=param.data.to(self.base_device),
                                                    requires_grad=param.requires_grad)
                                )
            self.log_sigma_theta.append(torch.nn.Parameter(
                                            data=torch.randn(size=param.shape)\
                                                .to(self.base_device) - 4,
                                            requires_grad=param.requires_grad)
                                        )
            self.log_v_q.append(torch.nn.Parameter(
                                    data=torch.randn(size=param.shape).to(self.base_device) - 4,
                                    requires_grad=param.requires_grad)
                                )

        self.gamma_p = torch.nn.Parameter(data=torch.tensor(float(gamma_p)).to(self.base_device))
        self.gamma_q = torch.nn.Parameter(data=torch.tensor(float(gamma_q)).to(self.base_device))

        self.inner_lr = torch.nn.Parameter(data=torch.tensor(float(inner_lr)).to(self.base_device))
        self.classifier_lr = torch.nn.Parameter(data=torch.tensor(float(classifier_lr))\
                                .to(self.base_device))

        # getting the key, parameter from initialize task head 
        init_kwargs = self.get_task_init_kwargs(language_head_init_method, n_classes=lm_head_n)
        self.lm_head_n = lm_head_n
        self.lm_head = TaskHead.initialize_task_head(task_type='classification',
                                                     method=language_head_init_method,
                                                     init_kwargs=init_kwargs)

        # loading in meta optimizer 
        self.meta_lr = float(meta_lr)
        if optimizer_type == 'adam': 
            self.optimizer = torch.optim.Adam(params=self.meta_params_iter(), lr=self.meta_lr)
        else:
            logger.exception(f"Invalid optimizer type: {optimizer_type}")
            raise Exception(f"Invalid optimizer type: {optimizer_type}")

        # hyper-param for trading off ce-loss and kl-loss
        self.kl_weight = float(kl_weight)

        # number of steps to perform in the inner loop
        self.num_inner_steps = int(num_inner_steps)
        
        # set flag to indicate if first-order approximation should be used (à la Reptile)
        if isinstance(use_first_order, str):
            use_first_order = eval(use_first_order)

        self.use_first_order = use_first_order

    ###### Helper functions ######

    def meta_params_iter(self):
        """ Returns an iterator over all of the meta parameters"""
        return itertools.chain(self.mu_theta, self.log_sigma_theta, self.log_v_q, 
                               [self.gamma_p, self.gamma_q, self.inner_lr, self.classifier_lr],
                               self.lm_head.values())

    def parameters(self):
        """ Overriding parent behavior to only return meta parameters """
        return self.meta_params_iter()

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


    def get_task_init_kwargs(self, task_init_method, data_batch=None, **kwargs):
        """ 
        Overrides the parent's method to also include kwargs for protomaml.

        Args:
            * task_init_method (str): Method for intiializing the task head
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure)

        Returns:
            * init_kwargs (dict): Keyword arguments used by the initialization function 
        """

        init_kwargs = super().get_task_init_kwargs(**kwargs)
        
        if 'protomaml' in task_init_method:
            assert(data_batch is not None),\
                "Use of protomaml as a classification head initializer requires a data_batch"

            init_kwargs['functional_model'] = self.functional_model
            init_kwargs['params'] = self.mu_theta
            init_kwargs['data_batch'] = data_batch

        return init_kwargs
    
    def functionalize_model(self):
        """ Helper function for converting base_model into a functionalized form"""
        self.functional_model = higher.patch.make_functional(module=self.base_model)

        # NOTE: these two lines are SUPER important (otherwise computation graph explodes)
        self.functional_model.track_higher_grads = False
        self.functional_model._fast_params = [[]]

    ###### Model adaptation methods ######

    """
    Platipus relies on these methods to adapt and sample from the weight distributions 
    that it meta-learned. 
    """

    def _sample_adapted_weights(self, data_batch, sampling_std, return_adapted_mean=False, 
                                device=None, **adaptation_kwargs): 
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
        if device is None:
            device = self.base_device

        adapted_params = self._adapt_params_classification(data_batch, **adaptation_kwargs)

        sampled_weights = [None] * len(adapted_params) 

        for i in range(len(sampled_weights)):
            if adapted_params[i].requires_grad: 
                sampled_weights[i] = adapted_params[i] + \
                    torch.randn_like(input=adapted_params[i], device=device) * \
                        torch.exp(input=sampling_std[i])
            else: 
                sampled_weights[i] = self.mu_theta[i]

        if return_adapted_mean:
            return (adapted_params, sampled_weights)
        else: 
            return sampled_weights  

    # adaptation to classification tasks (e.g. the meta-training task is a classification task)
    def _adapt_params_classification(self, data_batch, params, lm_head_weights, learning_rate,
                                     optimize_classifier=False, clone_params=True,
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
            * lm_head_weights (dict): Weights of the lm classification head
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
            
            # Running forward pass through functioanl model and computing classificaiton loss
            outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                    attention_mask=data_batch['attention_mask'],
                                                    params=adapted_params)

            _, loss = self._compute_task_loss(outputs, data_batch, lm_head_weights,
                                              task_type='classification')
                                                        
            # Computing resulting gradients of the inner loop
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
            # should only be done once we've already sampled phi weights
            if optimize_classifier:
                classifier_grads = torch.autograd.grad(
                    outputs=loss,
                    inputs=lm_head_weights.values(),
                )
                for idx, weight_name in enumerate(lm_head_weights.keys()):
                    lm_head_weights[weight_name] = lm_head_weights[weight_name] - \
                                                    self.classifier_lr * classifier_grads[idx]

        return adapted_params


    ###### Model training methods ######

    ### Multi Processing Helper Method
    def run_inner_loop_mp(self, rank, world_size, data_queue, loss_queue, step_optimizer, 
                          num_tasks_per_iteration):
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

        device, ddp = self.setup_DDP(rank, world_size)

        self.functionalize_model()

        while True: 
            # The main process sends signal to update optimizers

            while True: 
                # Waiting for the next batch of data 
                # NOTE: If there is no data either 1) the dataloading pipeline is taking a while 
                # or 2) the main process is waiting for all the workers to finish 
                try:
                    batch = data_queue.get(block=False)[0]
                    break
                except EmptyQueue: 
                    pass

                if step_optimizer.is_set():
                    # make sure all workers have taken an optimizer step
                    self.optimizer_step(set_zero_grad=True)
                    dist.barrier()

                    # once all workers have update params clear the flag to continue training
                    step_optimizer.clear()

                time.sleep(1) 

            task_name, support_batch, query_batch = batch

            task_loss, (ce_loss, kl_loss) = ddp(self, support_batch, query_batch, device)

            task_loss = task_loss/num_tasks_per_iteration
            ce_loss = ce_loss/num_tasks_per_iteration
            kl_loss = kl_loss/num_tasks_per_iteration

            task_loss.backward()

            loss_queue.put([[task_loss.detach().item(),
                            ce_loss.detach().item(),
                            kl_loss.detach().item()]])

    ### Main Inner Training Loop 
    def run_inner_loop(self, support_batch, query_batch, device=None, *args, **kwargs): 
        """
        Implements steps 6-11 outlined in the algorithm 1 of the Platipus paper
        https://arxiv.org/pdf/1806.02817.pdf, i.e. the inner loop optimization step.
        
        Args: 
            * support_batch: A dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch: Same as support_batch, but for the data of the query set 
            * device: Optional string to specify a device to override base_device

        Returns: 
            * loss (torch.Tensor): Loss value of the inner loop calculations
        """
        if device is None:
            device = self.base_device

        if not hasattr(self, "functional_model"):
            # If not using multiprocessing training, the first iteration of run_inner_loop
            # will have to functionalize the model 
            self.functionalize_model()

        self.functional_model.train()

        # Moving data to appropriate device
        support_batch = move_to_device(support_batch, device)
        query_batch = move_to_device(query_batch, device)
       
        # cloning LM head in order to be locally adapted
        adapted_lm_head = {key: torch.clone(param) for key, param in self.lm_head.items()}

        # sampling theta after adapting model to query_batch
        mu_theta_query, theta = self._sample_adapted_weights(query_batch, 
                                                             sampling_std=self.log_v_q,
                                                             return_adapted_mean=True,
                                                             params=self.mu_theta,
                                                             lm_head_weights=self.lm_head,
                                                             learning_rate=self.gamma_q,
                                                             clone_params=True,
                                                             device=device)

        # adapting theta to the support set -> adapted params are phi
        phi = self._adapt_params_classification(support_batch, 
                                                params=theta, 
                                                lm_head_weights=adapted_lm_head,
                                                learning_rate=self.inner_lr,
                                                clone_params=False,
                                                optimize_classifier=True)

        # evaluating on the query batch using the adapted params phi  
        self.functional_model.eval()

        outputs = self.functional_model.forward(input_ids=query_batch['input_ids'],
                                                attention_mask=query_batch['attention_mask'],
                                                params=phi)

        self.functional_model.train()

        _, ce_loss = self._compute_task_loss(outputs, query_batch, adapted_lm_head, 
                                             task_type='classification')

        # computing KL loss (requires adapting mu_theta to the support set)
        
        # mu theta adapted to the support set (line 10 from algorithm 1) 
        mu_theta_support = self._adapt_params_classification(support_batch, 
                                                             params=self.mu_theta,
                                                             lm_head_weights=self.lm_head,
                                                             learning_rate=self.gamma_p,
                                                             clone_params=True,)

        kl_loss = kl_divergence_gaussians(p=[*mu_theta_query, *self.log_v_q],
                                          q=[*mu_theta_support, *self.log_sigma_theta])

        loss = ce_loss + self.kl_weight * kl_loss

        return loss, (ce_loss, self.kl_weight * kl_loss)


    ###### Model evaluation methods ######

    # for NLU classification tasks

    def run_finetuning_classification(self, finetune_dataloader, n_classes, adaptation_batch,
                                      task_head_init_method, max_finetuning_batch_steps=-1,
                                      **kwargs): 
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
            * adaptation_batch (dict): A dictionary containing the required data for running
                a forward pass of the model (see run_inner_loop for explanation of the dictionary)
            * task_head_init_method (str): Method for initializing task head 
            * max_finetuning_batch_steps (int): Optional maximum number of batch steps to take 
                for model finetuning 

        Returns:
            * inference_params dict containing: 
                * finetuned_params ([nn.Parameter]): List of the finetuned model's parameters
                * task_head_weights (dict): Weights of task head (classifier head)
        """

        if not hasattr(self, "functional_model"):
            # If the model is only being evaluated it might not have a functionalized version 
            self.functionalize_model()

        self.functional_model.train()

        ### Sampling weights from mu_theta using the passed in adaptation_batch 
        # NOTE: the adaptation batch is in the same form as the batches of training used 
        # during meta training 
        adaptation_batch = move_to_device(adaptation_batch, self.base_device)
        sampled_theta = self._sample_adapted_weights(adaptation_batch,
                                                     sampling_std=self.log_sigma_theta,
                                                     params=self.mu_theta,
                                                     lm_head_weights=self.lm_head,
                                                     learning_rate=self.gamma_p,
                                                     clone_params=True,
                                                     evaluation_mode=True)

        ### Initializing the task head used for the downstream NLU task
        task_init_data_batch = move_to_device(next(iter(finetune_dataloader)), self.base_device)
        init_kwargs = self.get_task_init_kwargs(task_head_init_method, n_classes=n_classes,
                                                data_batch=task_init_data_batch)
        task_head_weights = TaskHead.initialize_task_head(task_type='classification',
                                                          method=task_head_init_method,
                                                          init_kwargs=init_kwargs)

        # detaching parameters from original computation graph to create new leaf variables
        finetuned_theta = []
        for p in sampled_theta:
            detached_p = p.detach()
            detached_p.requires_grad = p.requires_grad
            finetuned_theta.append(detached_p)

        finetuned_task_head_weights = {}
        for k, p in task_head_weights.items():
            detached_p = p.detach()
            detached_p.requires_grad = True
            finetuned_task_head_weights[k] = detached_p

        finetune_params = itertools.chain(finetuned_theta,
                                          finetuned_task_head_weights.values())
        finetune_optimizer = torch.optim.Adam(params=finetune_params)

        for batch_idx, data_batch in enumerate(finetune_dataloader):
            data_batch = move_to_device(data_batch, self.base_device)
            finetune_optimizer.zero_grad()

            # run SGD on the finetuned theta parameters

            outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                    attention_mask=data_batch['attention_mask'],
                                                    params=finetuned_theta)

            _, loss = self._compute_task_loss(outputs, data_batch, finetuned_task_head_weights, 
                                              task_type='classification')

            loss.backward()
            finetune_optimizer.step()

            if max_finetuning_batch_steps > 0 and (batch_idx + 1) >= max_finetuning_batch_steps:
                break

        inference_params = {
            "finetuned_params": finetuned_theta, 
            "task_head_weights": finetuned_task_head_weights
        }

        return inference_params


    def run_inference_classification(self, inference_dataloader, finetuned_params, 
                                     task_head_weights, adaptation_batch=None, **kwargs):
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
            * task_head_weights (dict): Weights of task head (classifier head)
            * adaptation_batch (dict): A dictionary containing the required data for running
                a forward pass of the model (see run_inner_loop for explanation of the dictionary)

        Returns: 
            * predictions ([int]): A dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): The value of the classification loss on the inference dataset.
        """
        if not hasattr(self, "functional_model"):
            # If the model is only being evaluated (and not being finetuned) it might not have
            # a functionalized version
            self.functionalize_model()

        if adaptation_batch is not None:
                # if adaptation_batch is passed in, we adapt the model's parameters to this data
                adaptation_batch = move_to_device(adaptation_batch, self.base_device)

                finetuned_params = self._sample_adapted_weights(adaptation_batch, 
                                                                sampling_std=self.log_sigma_theta,
                                                                params=finetuned_params,
                                                                lm_head_weights=self.lm_head,
                                                                learning_rate=self.gamma_p,
                                                                clone_params=True,
                                                                evaluation_mode=True)
        
        predictions = []
        total_loss = 0.0
        total_samples = 0

        # Running final inference script over the evaluation data
        with torch.no_grad():
            self.functional_model.eval()

            for data_batch in inference_dataloader: 
                data_batch = move_to_device(data_batch, self.base_device)

                outputs = self.functional_model.forward(input_ids=data_batch['input_ids'],
                                                        attention_mask=data_batch['attention_mask'],
                                                        params=finetuned_params)

                logits, loss = self._compute_task_loss(outputs, data_batch, task_head_weights,
                                                       task_type='classification')

                predictions.extend(torch.argmax(logits, dim=-1).tolist())

                batch_size = logits.size(0)
                total_loss += loss.item() * batch_size # loss is averaged across batch
                total_samples += batch_size 

            total_loss /= total_samples

            self.functional_model.train()

        return (predictions, total_loss)
