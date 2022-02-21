__author__ = 'Richard Diehl Martinez'
'''Wrapper class for training and evaluating a model using meta learning '''

import typing
import logging 

from .models import XLMR
from .metalearners import Platipus
from .utils import device as DEFAULT_DEVICE, move_to_device

logger = logging.getLogger(__name__)

class Problyglot(object):
    """
    Orchestrates model loading, training and evaluation using a specific 
    type of (meta-)learner.
    """

    def __init__(self, config) -> None:
        ''' Initialize base model and meta learning method'''

        # config params need to be accessed by several methods
        self.config = config

        self.device = config.get("PROBLYGLOT", "device", fallback=DEFAULT_DEVICE)
        logger.info(f"Running problyglot on device: {self.device}")

        base_model_name = config.get("BASE_MODEL", "name")
        self.base_model = self.load_model(base_model_name)
        
        learner_method = config.get("LEARNER", "method")
        self.learner = self.load_learner(learner_method)

        self.num_tasks_per_iteration = config.getint("PROBLYGLOT", "num_tasks_per_iteration", fallback=1)

    def load_model(self, base_model_name):
        """Helper function for reading in base model - should be intialized with from_kwargs() class method """
        logger.info(f"Loading base model: {base_model_name}")
        model = None
        model_kwargs = dict(self.config.items("BASE_MODEL"))
        if base_model_name == 'xlm_r':
            model_cls = XLMR
        else:
            raise Exception(f"Invalid base model type: {base_model_name}")

        model = model_cls.from_kwargs(**model_kwargs)
        model.to(self.device)

        logger.debug("Base Model Architecture: ")
        logger.debug(model)

        return model

    def load_learner(self, learner_method):
        """Helper function for reading in (meta) learning procedure"""
        logger.info(f"Using learner: {learner_method}")
        learner = None
        learner_kwargs = dict(self.config.items("LEARNER"))
        # NOTE: if any of the learners' params need to be on the GPU the learner class should take care of 
        # moving these params over during initialization
        if learner_method == 'platipus':
            learner = Platipus(self.base_model, device=self.device, **learner_kwargs)
        else:
            raise Exception(f"Invalid learner method: learner_method")

        return learner

    def train(self, train_dataloader) -> None: 
        """ 
        Train the self.base_model via the self.learner training procedure 
        on data stored in train_dataloader
        """

        num_task_batches = 0 # counter tracks number of batches of tasks seen by metalearner

        for batch_idx, batch in enumerate(train_dataloader):
            task_name, support_batch, query_batch = batch

            support_batch = move_to_device(support_batch, self.device)
            query_batch = move_to_device(query_batch, self.device)

            print(f"got data for task: {task_name}")

            task_batch_loss = 0. # training loss on the batch of tasks
            for task_idx in range(self.num_tasks_per_iteration): 
                print("running inner loop")
                task_loss = self.learner.run_inner_loop(support_batch, query_batch)
                task_loss = task_loss / self.num_tasks_per_iteration # normalizing loss 
                task_loss.backward()
                task_batch_loss += task_loss.detach().item()

            num_task_batches += 1
            self.learner.optimizer_step(set_zero_grad=True)
            

            # TODO: Logging results 
            # TODO: Checkpointing 
