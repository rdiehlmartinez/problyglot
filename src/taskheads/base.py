__author__ = 'Richard Diehl Martinez'
""" Base class for task heads """

import abc 
import logging

from collections import defaultdict

logger = logging.getLogger(__name__)

class TaskHead(object, metaclass=abc.ABCMeta):
    """
    Everytime we train and evaluate a model using a learner, we need to initialize 
    and train a 'task head'. For every type of task (e.g. classification, q&a) we can have 
    different methods for initializing the task head (e.g. randomly). To initialize a task 
    head, we first define an initialization function that we wrap using the 
    register_initialization_method() method, and then we can call initialize_task_head() with
    the appropriate parameters.
    """

    _task_head_initializers = defaultdict(dict)

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """ Implements the functional forward pass through the task head"""
        raise NotImplementedError()

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
            cls._task_head_initializers[task_type][method] = initialization_function
        else:
            error_msg = f"""Could not register task head initializer:
                                {initialization_function.__name__},
                                the name of the function should be of the form:
                                (task_type)_(initialization_method).
                                E.g.: classification_random(...)
                         """
            logger.exception(error_msg)
            raise Exception(error_msg)

        return initialization_function

    @classmethod
    def initialize_task_head(cls, task_type, method, init_kwargs): 
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
            * task_head_weights (dict): An arbitrary dictionary containing weights for 
                the initialized task head. Depending on the task_type the weights returned 
                might be different. 
        """

        try: 
            initialization_function = cls._task_head_initializers[task_type][method]
        except KeyError:
            logger.exception("Could not initialize task head; invalid task type or method")
            raise Exception("Could not initialize task head; invalid task type or method")
        
        return initialization_function(**init_kwargs)
        