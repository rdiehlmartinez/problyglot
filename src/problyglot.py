__author__ = 'Richard Diehl Martinez'
''' Wrapper class for training and evaluating a model using a given meta learning technique '''

import typing
import logging 
import wandb

from .models import XLMR
from .metalearners import Platipus
from .evaluation import Evaluator
from .utils import device as DEFAULT_DEVICE, move_to_device
from .datasets import MetaDataset, MetaDataLoader

logger = logging.getLogger(__name__)

class Problyglot(object):
    """
    Orchestrates model loading, training and evaluation using a specific 
    type of (meta-)learner.
    """

    def __init__(self, config) -> None:
        """ Initialize base model and meta learning method """
        # setting up meta dataset for training if provided in config
        if 'META_DATASET' in config:
            self.meta_dataset = MetaDataset(config)
            self.meta_dataloader = MetaDataLoader(self.meta_dataset)

        # config params need to be accessed by several methods
        self.config = config

        # Setting device 
        self.device = config.get("PROBLYGLOT", "device", fallback=DEFAULT_DEVICE)
        logger.info(f"Running problyglot on device: {self.device}")

        # setting base model 
        base_model_name = config.get("BASE_MODEL", "name")
        self.base_model = self.load_model(base_model_name)
        
        # setting learner 
        learner_method = config.get("LEARNER", "method")
        self.learner = self.load_learner(learner_method)

        # setting up metrics for logging of training 
        wandb.define_metric("num_task_batches")
        wandb.define_metric("train*", step_metric="num_task_batches")

        # setting evaluator 
        if 'EVALUATION' in config:
            self.evaluator = Evaluator(config)
        
        # setting problyglot specific configurations
        self.num_tasks_per_iteration = config.getint("PROBLYGLOT", "num_tasks_per_iteration", fallback=1)
        self.eval_every_n_iteration = config.getint("PROBLYGLOT", "eval_every_n_iteration", fallback=0)


    def load_model(self, base_model_name):
        """ Helper function for reading in base model - should be intialized with from_kwargs() class method """

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
        """ Helper function for reading in (meta) learning procedure """

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

    def train(self) -> None: 
        """ 
        Train the self.base_model via the self.learner training procedure 
        on data stored in self.meta_dataloader
        """
        logger.info("Beginning to train model")

        # counter tracks number of batches of tasks seen by metalearner
        num_task_batches = 0 

        # counter tracks loss over an entire batch of tasks  
        task_batch_loss = 0 

        for batch_idx, batch in enumerate(self.meta_dataloader):
            task_name, support_batch, query_batch = batch
            logger.debug(f"\t Training on task idx {batch_idx} - task: {task_name}")

            support_batch = move_to_device(support_batch, self.device)
            query_batch = move_to_device(query_batch, self.device)

            task_loss = self.learner.run_inner_loop(support_batch, query_batch)            
            task_loss = task_loss / self.num_tasks_per_iteration # normalizing loss 
            task_loss.backward()
            task_batch_loss += task_loss.detach().item()

            if ((batch_idx + 1) % self.num_tasks_per_iteration == 0):
                
                num_task_batches += 1

                self.learner.optimizer_step(set_zero_grad=True)
                logger.info(f"No. batches of tasks processed: {num_task_batches} -- Task batch loss: {task_batch_loss}")
                wandb.log({"train": {"loss": task_batch_loss}, "num_task_batches": num_task_batches})
                task_batch_loss = 0 

                # possibly run evaluation of the model
                if (self.eval_every_n_iteration and num_task_batches % self.eval_every_n_iteration == 0):
                    self.evaluator.run(self.learner, num_task_batches=num_task_batches)                 

                if num_task_batches == 2:
                    self.meta_dataset.shutdown()
                    exit()  

                # possibly save a checkpoint of the model 
                # TODO


        logger.info("Finished training model")
        logger.info("Shutting down meta dataloader workers")
        self.meta_dataset.shutdown()