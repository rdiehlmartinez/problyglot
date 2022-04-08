__author__ = 'Richard Diehl Martinez'
''' Wrapper class for training and evaluating a model using a given meta learning technique '''

import typing
import torch
import logging 
import wandb
import os 

from .models import XLMR
from .metalearners import Platipus, BaselineLearner
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
        
        # config params need to be accessed by several methods
        self.config = config
        
        # whether to log out information to w&b
        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)

        # setting up meta dataset for training if provided in config
        if 'META_DATASET' in config:
            self.meta_dataset = MetaDataset(config)
            self.meta_dataloader = MetaDataLoader(self.meta_dataset)

        # Setting device 
        self.device = config.get("PROBLYGLOT", "device", fallback=DEFAULT_DEVICE)
        logger.info(f"Running problyglot on device: {self.device}")

        # setting base model 
        base_model_name = config.get("BASE_MODEL", "name")
        self.base_model = self.load_model(base_model_name)
        
        # setting learner 
        learner_method = config.get("LEARNER", "method")
        self.learner = self.load_learner(learner_method)

        if self.use_wandb:
            # setting up metrics for logging to wandb
            wandb.define_metric("num_task_batches")

        # setting evaluator 
        if 'EVALUATION' in config:
            self.evaluator = Evaluator(config)

    def load_model(self, base_model_name):
        """
        Helper function for reading in base model, should be intialized with the 
        from_kwargs() class method 
        """

        logger.info(f"Loading base model: {base_model_name}")
        model = None
        model_kwargs = dict(self.config.items("BASE_MODEL"))
        if base_model_name == 'xlm_r':
            model_cls = XLMR
        else:
            logger.exception(f"Invalid base model type: {base_model_name}")
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
        # NOTE: if any of the learners' params need to be on the GPU the learner class should
        # take care of moving these params over during initialization
        if learner_method == 'platipus':
            learner_cls = Platipus
        elif learner_method == 'baseline': 
            learner_cls = BaselineLearner
        else:
            logger.exception(f"Invalid learner method: {learner_method}")
            raise Exception(f"Invalid learner method: {learner_method}")

        learner = learner_cls(self.base_model, device=self.device, **learner_kwargs)

        # possibly load in learner checkpoint
        checkpoint_file = self.config.get("LEARNER", "checkpoint_file", fallback="")
        if checkpoint_file:
            if not self.use_wandb:
                logger.warning("Could not load in checkpoint file, use_wandb is set to False")
            else:
                checkpoint_run = self.config.get("LEARNER", "checkpoint_run")
                logger.info(f"Loading in checkpoint file: {checkpoint_file}")
                wandb_checkpoint = wandb.restore(checkpoint_file, run_path=checkpoint_run)
                checkpoint = torch.load(wandb_checkpoint.name)
                learner.load_state_dict(checkpoint['learner_state_dict'], strict=False)
                learner.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                os.rename(os.path.join(wandb.run.dir, checkpoint_file),
                          os.path.join(wandb.run.dir, "loaded_checkpoint.pt"))
        else:
            logger.info("No checkpoint used - learning from scratch")

        return learner

    def __call__(self) -> None: 
        """ 
        Train the self.base_model via the self.learner training procedure 
        on data stored in self.meta_dataloader
        """

        # Evaluation Mode - will return early
        if not hasattr(self, "meta_dataset"):
            logger.info("Running problygot in evaluation mode")

            if not hasattr(self, "evaluator"):
                logger.warning("No evaluator specified - running problygot is a no-op")
                return 
            
            logger.info("Finished running evaluation model")
            self.evaluator.run(self.learner)
            return 

        # Training Mode
        logger.info("Running problyglot in training mode")

        # setting problyglot training configurations
        num_tasks_per_iteration = self.config.getint("PROBLYGLOT", "num_tasks_per_iteration",
                                                     fallback=1)
        eval_every_n_iteration = self.config.getint("PROBLYGLOT", "eval_every_n_iteration",
                                                    fallback=0)
        max_task_batch_steps = self.config.getint("PROBLYGLOT", "max_task_batch_steps",
                                                  fallback=1)

        # counter tracks number of batches of tasks seen by metalearner
        num_task_batches = 0 

        # counter tracks loss over an entire batch of tasks  
        task_batch_loss = 0 

        # metric for logging training data
        if self.use_wandb:
            wandb.define_metric("train.loss", step_metric="num_task_batches", summary='min')

        if self.config.getboolean("PROBLYGLOT", "run_initial_eval", fallback=True):
            logger.info("Initial evaluation before model training")
            self.evaluator.run(self.learner, num_task_batches=0)

        logger.info("Starting model training")
        for batch_idx, batch in enumerate(self.meta_dataloader):
            task_name, support_batch, query_batch = batch
            logger.debug(f"\t Training on task idx {batch_idx} - task: {task_name}")

            support_batch = move_to_device(support_batch, self.device)
            query_batch = move_to_device(query_batch, self.device)

            task_loss = self.learner.run_inner_loop(support_batch, query_batch)    
        
            task_loss = task_loss / num_tasks_per_iteration # normalizing loss 
            task_loss.backward()

            task_batch_loss += task_loss.detach().item()

            if ((batch_idx + 1) % num_tasks_per_iteration == 0):
                
                num_task_batches += 1

                self.learner.optimizer_step(set_zero_grad=True)
                logger.info(f"No. batches of tasks processed: {num_task_batches}")
                logger.info(f"\t(Meta) training loss: {task_batch_loss}")
                if self.use_wandb:
                    wandb.log({"train": {"loss": task_batch_loss},
                               "num_task_batches": num_task_batches})
                task_batch_loss = 0 

                # possibly run evaluation of the model
                if (eval_every_n_iteration and num_task_batches % eval_every_n_iteration == 0):
                    self.evaluator.run(self.learner, num_task_batches=num_task_batches)

                if (num_task_batches % max_task_batch_steps == 0):
                    break
                
        logger.info("Finished training model")
        if self.config.getboolean('PROBLYGLOT', 'save_final_model', fallback=True):
            if not self.use_wandb:
                logger.error("Cannot save model checkpoint because use_wandb set to False")
            else:
                logger.info(f"Saving trained model")
                checkpoint = {
                    'learner_state_dict': self.learner.state_dict(),
                    'optimizer_state_dict': self.learner.optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(wandb.run.dir, f"final.pt"))
        logger.info("Shutting down meta dataloader workers")
        self.meta_dataset.shutdown()