__author__ = 'Richard Diehl Martinez'
''' Wrapper class for training and evaluating a model using a given meta learning technique '''

import typing
import torch
import logging 
import wandb
import time 
import os 

import torch.multiprocessing as mp

from .models import XLMR
from .metalearners import Platipus, BaselineLearner
from .evaluation import Evaluator
from .utils import device as DEFAULT_DEVICE, move_to_device, num_gpus
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
        self.base_device = config.get("PROBLYGLOT", "device", fallback=DEFAULT_DEVICE)
        logger.info(f"Running problyglot on device: {self.base_device}")

        # setting base model 
        self.base_model_name = config.get("BASE_MODEL", "name")
        self.base_model = self.load_model(self.base_model_name)

        # setting learner 
        self.learner_method = self.config.get("LEARNER", "method")
        self.learner = self.load_learner(self.learner_method)
        
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

        logger.debug("Base Model Architecture: ")
        logger.debug(model)

        return model

    def load_learner(self, learner_method):
        """ Helper function for reading in (meta) learning procedure """

        logger.info(f"Using learner: {learner_method}")
        learner = None
        learner_kwargs = dict(self.config.items("LEARNER"))

        if learner_method == 'platipus':
            learner_cls = Platipus
        elif learner_method == 'baseline': 
            learner_cls = BaselineLearner
        else:
            logger.exception(f"Invalid learner method: {learner_method}")
            raise Exception(f"Invalid learner method: {learner_method}")

        learner = learner_cls(self.base_model, base_device=self.base_device, **learner_kwargs)

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

        ### --------- Evaluation Mode (will return early) ---------- 
        if not hasattr(self, "meta_dataset"):
            logger.info("Running problygot in evaluation mode")

            if not hasattr(self, "evaluator"):
                logger.warning("No evaluator specified - running problygot is a no-op")
                return 
            
            logger.info("Finished running evaluation model")
            self.evaluator.run(self.learner)
            return 


        ### --------- Training Mode ---------- 

        logger.info("Running problyglot in training mode")

        ### reading in training configs

        num_tasks_per_iteration = self.config.getint("PROBLYGLOT", "num_tasks_per_iteration",
                                                     fallback=1)
        eval_every_n_iteration = self.config.getint("PROBLYGLOT", "eval_every_n_iteration",
                                                    fallback=0)
        max_task_batch_steps = self.config.getint("PROBLYGLOT", "max_task_batch_steps",
                                                  fallback=1)

        ### If using n GPUs we launch n processes that run the run_inner_loop_mp function 

        # If using multiple GPUs
        if num_gpus > 1:
            
            if num_tasks_per_iteration % num_gpus != 0:
                error_msg = "Num tasks per iteration has to be dividable by num_pus!"
                logger.exception(error_msg)
                raise Exception(error_msg)

            logger.info(f"Running data parallel training with {num_gpus} workers")
            spawn_context = mp.get_context('spawn')

            data_queue = spawn_context.Queue()
            loss_queue = spawn_context.Queue()

            gpu_workers = []
            for rank in range(num_gpus):
                p = spawn_context.Process(target=self.learner.run_inner_loop_mp,
                                          args=(rank, num_gpus, data_queue, loss_queue,
                                                num_tasks_per_iteration  
                                               )
                                          )
                p.start()
                gpu_workers.append(p)


        ### Setting up tracking variables and w&b metrics  

        # counter tracks number of batches of tasks seen by metalearner
        num_task_batches = 0 

        # counter tracks loss over an entire batch of tasks  
        task_batch_loss = 0 
        if self.learner_method == "platipus":
            # platipus tracks the ce and kl parts of the loss function
            task_batch_ce_loss = 0
            task_batch_kl_loss = 0

        # metric for logging training data
        if self.use_wandb:
            wandb.define_metric("train.loss", step_metric="num_task_batches", summary='min')

            if self.learner_method == "platipus":
                # platipus tracks additional information about the parts of the loss functions,
                # and the learning procedure of the hyper-parameters
                wandb.define_metric("train.loss_ce", step_metric="num_task_batches", summary='min')
                wandb.define_metric("train.loss_kl", step_metric="num_task_batches", summary='min')

                wandb.define_metric("gamma_p", step_metric="num_task_batches")
                wandb.define_metric("gamma_q", step_metric="num_task_batches")
                wandb.define_metric("classifier_lr", step_metric="num_task_batches")
                wandb.define_metric("inner_lr", step_metric="num_task_batches")

        if self.config.getboolean("PROBLYGLOT", "run_initial_eval", fallback=True):
            logger.info("Initial evaluation before model training")
            self.evaluator.run(self.learner, num_task_batches=0)


        ### Model training loop

        logger.info("Starting model training")
        for batch_idx, batch in enumerate(self.meta_dataloader):
            if num_gpus > 1:
                ## Filling up data queue for workers to process
                data_queue.put([batch], False)
            else:
                ## Basic training with just a single GPU 
                task_name, support_batch, query_batch = batch
                logger.debug(f"\t Training on task: {task_name}")

                support_batch = move_to_device(support_batch, self.base_device)
                query_batch = move_to_device(query_batch, self.base_device)

                if self.learner_method == "platipus":
                    task_loss, (ce_loss, kl_loss) = self.learner.run_inner_loop(support_batch,
                                                                                query_batch)
                else: 
                    task_loss = self.learner.run_inner_loop(support_batch, query_batch)
            
                task_loss = task_loss / num_tasks_per_iteration # normalizing loss 
                if self.learner_method == "platipus":
                    ce_loss = ce_loss/num_tasks_per_iteration
                    kl_loss = kl_loss/num_tasks_per_iteration
            
                task_loss.backward()

                task_batch_loss += task_loss.detach().item()
                if self.learner_method == "platipus":
                    task_batch_ce_loss += ce_loss.detach().item()
                    task_batch_kl_loss += kl_loss.detach().item()

            if ((batch_idx + 1) % num_tasks_per_iteration == 0):
                if num_gpus > 1: 

                    while True:
                        # Simple but semi-inefficient way of waiting for all processes to finish
                        # computing gradients
                        time.sleep(2)
                        if loss_queue.qsize() == num_tasks_per_iteration:
                            break

                    ## Multi GPU: gathering up all of the task losses
                    for _ in range(num_tasks_per_iteration):

                        loss = loss_queue.get()[0]

                        # loss should already be normalized
                        if self.learner_method == "platipus":
                            task_loss, ce_loss, kl_loss = loss
                            task_batch_ce_loss += ce_loss
                            task_batch_kl_loss += kl_loss
                        else:
                            task_loss = loss
                        
                        task_batch_loss += task_loss
                                                
                # NOTE: We just finished one global meta task batch (batch of tasks)
                num_task_batches += 1

                self.learner.optimizer_step(set_zero_grad=True)
                logger.info(f"No. batches of tasks processed: {num_task_batches}")
                logger.info(f"\t(Meta) training loss: {task_batch_loss}")
                if self.use_wandb:
                    if self.learner_method == "platipus": 
                        # wandb logging additional info for platipus
                        wandb.log({"train.loss_ce": task_batch_ce_loss,
                                   "train.loss_kl": task_batch_kl_loss,
                                   "gamma_p": self.learner.gamma_p.item(),
                                   "gamma_q": self.learner.gamma_q.item(),
                                   "classifier_lr": self.learner.classifier_lr.item(), 
                                   "inner_lr": self.learner.inner_lr.item()},
                                   commit=False
                                  )

                    wandb.log({"train.loss": task_batch_loss,
                               "num_task_batches": num_task_batches},
                             )

                task_batch_loss = 0 
                if self.learner_method == "platipus":
                    task_batch_ce_loss = 0
                    task_batch_kl_loss = 0

                # possibly run evaluation of the model
                if (eval_every_n_iteration and num_task_batches % eval_every_n_iteration == 0):
                    self.evaluator.run(self.learner, num_task_batches=num_task_batches)

                if (num_task_batches % max_task_batch_steps == 0):
                    # NOTE: stop training if we've done max_task_batch_steps global update steps
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