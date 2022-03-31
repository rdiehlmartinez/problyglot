__author__ = 'Richard Diehl Martinez'
''' Base ABC Class for (meta) learners '''

import abc 
import torch
import math

class BaseLearner(torch.nn.Module, metaclass=abc.ABCMeta):

    def __init__(self):
        """
        BaseLearner establishes the inferface for the learner class and 
        inherents from torch.nn.Module
        """
        super().__init__()

    ###### Helper methods ######

    _classification_head_initializers = {}

    @classmethod
    def register_initialization_method(cls, initialization_function):
        """ 
        Decorator function that takes in an initialization function and registers this 
        as a function to use for intitializing the weights of a task head.
        Args: 
            * initialization_function (type.function): A function that initializes task 
                head parameters 
        """
        task_type, method = initialization_function.__name__.split('_', 1)

        if task_type == 'classification': 
            cls._classification_head_initializers[method] = initialization_function
        else:
            raise Exception(f"""Could not register task head initializer:
                                {initialization_function.__name__},
                                the name of the function should be of the form:
                                (task_type)_(initialization_method).
                                E.g.: classification_random(...)
                            """
            )

        return initialization_function

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

    def initialize_task_head(self, task_type, method, init_kwargs): 
        """
        Method for initializing the weights of a task head. Reads in two strings, task_type and
        method which jointly specify the task head and type of initialization method to use. 
        Then calls on the corresponding function and forwards the **init_kwargs keyword arguments. 
        
        Args: 
            * task_type (str): Type of task head (e.g. 'classification')
            * method (str): Method to use for initializing the weights of the task head 
                (e.g. 'random')
            * init_kwargs (dict): Keyword arguments used by the initialization function 
        Returns: 
            * task_classifier_weights (dict): An arbitrary dictionary containing weights for 
                the initialized task head. Depending on the task_type the weights returned 
                might be different. 
        """

        if task_type == 'classification':
            initialization_function = self._classification_head_initializers[method]
        else:
            raise Exception(f"Could not initialize task head - unknown task type: {task_type}")
        
        return initialization_function(**init_kwargs)

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
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * input_target_idx (torch.tensor): Tensor indicating for each sample at what index
                    we apply the final classification layer 
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are
                    not pad tokens
            * query_batch [optional]: same as support_batch, but for the data of the query set.
                This argument might be optional depending on how the learner is implemented.

        Returns: 
            * loss (torch.tensor): a tensor containing the loss that results from the inner loop 
        """
        raise NotImplementedError()

    ###### Model evaluation methods ######

    @abc.abstractmethod
    def run_finetuning_classification(self, finetune_dataloader, *args, **kwargs):
        """
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
        Args: 
            * inference_dataloader (torch.data.Dataloader): The dataset for inference is passed
                in as a dataloader (in most cases this will be an NLUDataloader)

        Returns: 
            * predictions ([int]): a dictionary storing the model's predictions for each 
                datapoint passed in from the inference_dataloader as an int. 
            * loss (int): the value of the classification loss on the inference dataset.
        """
        raise NotImplementedError()


@BaseLearner.register_initialization_method
def classification_random(base_model_hidden_dim, n_classes, device, **kwargs):
    """
    Initializes classification task head using a random Xavier-He initialization method.

    Args: 
        * base_model_hidden_dim (int): The hidden dimensions of the outputs of the base_model 
        * n_classes (int): Number of classes to classify over 
        * device (str): Device type ('cuda' or 'cpu')
    Returns: 
        * task_classifier_weights (dict): {
            * weight -> (torch.Tensor): classification weight matrix
            * bias -> (torch.Tensor): classification bias vector
            }
    """
    # Xavier normal weight implementation
    std_weight = math.sqrt(2.0 / float(base_model_hidden_dim + n_classes))
    std_bias = math.sqrt(2.0 / float(n_classes))

    # weights need to be shape (out_features, in_features) to be compatible with linear layer
    classifier_weight = torch.randn((n_classes, base_model_hidden_dim), device=device) \
                            * std_weight
    classifier_bias = torch.randn((n_classes), device=device) * std_bias

    classifier_weight.requires_grad = True
    classifier_bias.requires_grad = True

    task_classifier_weights = { 
        "weight": classifier_weight,
        "bias": classifier_bias
    }

    return task_classifier_weights
