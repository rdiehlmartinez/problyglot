__author__ = 'Richard Diehl Martinez'
""" Deals with the orchestration of model evaluation on a variety of NLU tasks """

import logging
import os
import torch
import wandb
import numpy as np

from collections import defaultdict
from ..datasets import NLUDataLoader, NLU_DATASET_GENERATOR_MAPPING

logger = logging.getLogger(__name__)

TASK_PARAMS = {
    "xnli": {
        "task_type": "classification",
        "n_labels": 3, 
    }
}

class Evaluator(object): 
    def __init__(self, config): 
        """ 
        Sets up dataset generators for each eval task provided in the config. The 
        Evaluator class is a general framework for calling the inference procedures
        of a learner, and computing relevant metrics for each task that is meant to 
        be evaluated on.
        """

        ### read in and initialize dataset_generators 
        tasks_str = config.get("EVALUATION", "tasks", fallback="")
        if tasks_str == "":
            logger.warning("Initializing evaluator with no tasks for evaluation")

        self.tasks = tasks_str.split(',')

        self.dataset_generators = {task: NLU_DATASET_GENERATOR_MAPPING[task](config)
                                    for task in self.tasks}

        ### additional config setup

        self.batch_size = config.getint("EVALUATION", "batch_size", fallback=32)

        # maximum number of batch steps to finetune model on - users can pass in a list of 
        # comma-separated values or a single value. If not passed in finetunes on all avail data.
        max_finetuning_batch_steps_str = config.get("EVALUATION", "max_finetuning_batch_steps", 
                                                    fallback="")

        self.max_finetuning_batch_steps_list = []
        for num_steps in max_finetuning_batch_steps_str.split(','):
            processed_num_steps = int(num_steps)
            self.max_finetuning_batch_steps_list.append(processed_num_steps)
        
        if len(self.max_finetuning_batch_steps_list) == 0:
            self.max_finetuning_batch_steps_list.append(-1)


        self.save_eval_checkpoints = config.getboolean("EVALUATION", "save_eval_checkpoints",
                                                       fallback=False)
        if self.save_eval_checkpoints:
            # possibly keep track of previous runs of the evaluator for checkpoint purposes
            self.eval_run_tracker = defaultdict(list)

        self.save_latest_checkpoint = config.getboolean("EVALUATION", "save_latest_checkpoint",
                                                        fallback=True)

        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)
        
        self.learner_method = config.get("LEARNER", "method")


    ### Helper methods for computing evaluation metrics

    @staticmethod
    def compute_accuracy(predictions, evaluation_dataloader):
        """ Computes accuracy of predictions for the data of the evaluation_dataloader """
        labels = []
        for data_batch in evaluation_dataloader:
            labels.extend(data_batch['label_ids'].tolist())
        
        accuracy = (np.array(predictions) == np.array(labels)).sum()/len(labels)
        return accuracy

    ### Entry point to running evaluation

    def run(self, learner, num_task_batches=0):
        """ 
        Runs evaluation of the passed in learner on the self.tasks evaluation tasks. 
        Loops over each of the evaluation tasks in self.tasks and for each task 
        runs the learner's finetuning procedure and inference procedure. The inference 
        procedure returns some predictions which are then used to compute metrics for each
        of the tasks. 

        Args:
            * learner (subclass of BaseLearner): learning procedure 
            * num_task_batches (int): optional value of the current task batch number 
                at which we are evaluating
        """

        logger.info("")
        logger.info("-"*30)
        logger.info("Running evaluator")

        mark_best_ckpt = False

        for idx, task in enumerate(self.tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {task}")

            ## Setting up params for the given task
            task_params = TASK_PARAMS[task]
            dataset_generator = self.dataset_generators[task]

            if task_params['task_type'] == "classification": 
                compute_metric = self.compute_accuracy
                metric_name = "acc"
                metric_summary = 'max'
            else: 
                logger.exception(f"Invalid task type: {task_params['task_type']} for task: {task}")
                raise Exception(f"Invalid task type: {task_params['task_type']} for task: {task}")

            task_metrics = defaultdict(list)
            task_losses = defaultdict(list)


            for num_steps in self.max_finetuning_batch_steps_list:

                for subtask_idx, (finetune_dataset, evaluation_dataset) in enumerate(dataset_generator):
                    finetune_lng = finetune_dataset.language
                    evaluation_lng = evaluation_dataset.language
                    logger.info(f"\t Finetuning on: {finetune_lng} - evaluating on: {evaluation_lng}")

                    finetune_dataloader = NLUDataLoader(finetune_dataset,
                                                        batch_size=self.batch_size)
                    evaluation_dataloader = NLUDataLoader(evaluation_dataset,
                                                        batch_size=self.batch_size)

                    ### Running Finetuning
                    # Calling on run_finetuning returns a set of finetuned-parameters
                    finetune_adaptation_batch = None
                    task_head_init_method = dataset_generator.task_head_init_method

                    if dataset_generator.use_few_shot_adaptation or subtask_idx == 0:
                        if self.learner_method == "platipus":
                            finetune_adaptation_batch = finetune_dataset.get_adaptation_batch()
                        inference_params = learner.run_finetuning(
                                            finetune_dataloader=finetune_dataloader,
                                            adaptation_batch=finetune_adaptation_batch,
                                            task_head_init_method=task_head_init_method,
                                            max_finetuning_batch_steps=num_steps,
                                            **task_params)

                    ### Running Inference 
                    eval_adaptation_batch = None
                    if dataset_generator.adapt_on_eval:
                        if self.learner_method != "platipus":
                            logger.warning("(ignoring adapt_on_eval) - learner is not 'platipus'")
                        else:    
                            eval_adaptation_batch = evaluation_dataset.get_adaptation_batch()
            
                    predictions, eval_loss = learner.run_inference(
                                                        inference_dataloader=evaluation_dataloader,
                                                        adaptation_batch=eval_adaptation_batch,
                                                        **inference_params,
                                                        **task_params)

                    ### Logging out metrics
                    if self.use_wandb:
                        wandb.define_metric(f"{task}.{evaluation_lng}.{num_steps}.{metric_name}",
                                            step_metric="num_task_batches", summary=metric_summary)
                        wandb.define_metric(f"{task}.{evaluation_lng}.{num_steps}.loss",
                                            step_metric="num_task_batches", summary='min')

                    # compute metrics using predictions 
                    metric = compute_metric(predictions, evaluation_dataloader)
                    logger.info(f"\t \t Finetune steps: {num_steps} - " +\
                                f"{metric_name}: {metric:.4f} - Eval Loss: {eval_loss:.4f}")
                    if self.use_wandb:
                        wandb.log({task: {
                                        evaluation_lng: {
                                            num_steps: {
                                                "loss": eval_loss,
                                                metric_name: metric,
                                            },
                                        },
                                    },
                                "num_task_batches": num_task_batches
                                })
            
                    task_metrics[num_steps].append(metric)
                    task_losses[num_steps].append(eval_loss)
                
            
            for num_steps in self.max_finetuning_batch_steps_list: 
                # for each max finetune steps setting compute the average metrics and loss
                task_metric = task_metrics[num_steps]
                task_loss = task_losses[num_steps]

                task_metric_mean = sum(task_metric)/len(task_metric)
                task_loss_mean = sum(task_loss)/len(task_loss)

                if self.use_wandb:
                    wandb.define_metric(f"{task}.{num_steps}.{metric_name}",
                                        step_metric="num_task_batches",
                                        summary=metric_summary)
                    wandb.define_metric(f"{task}.{num_steps}.loss", step_metric="num_task_batches",
                                        summary='min')

                    wandb.log({task: {
                                num_steps:{
                                    "loss": task_loss_mean,
                                    metric_name: task_metric_mean,      
                                }
                            },
                            "num_task_batches": num_task_batches
                            })

                logger.info(f"\t (Task {idx}) Fintune steps: {num_steps} - " +\
                            f"Avg. {metric_name}: {task_metric_mean:.4f}")
                logger.info(f"\t (Task {idx}) Fintune steps: {num_steps} - " +\
                            f"Avg. Loss: {task_loss_mean:.4f}")

                # If we are saving eval checkpoints, then do some book-keeping to keep track of
                # the best model
                if self.save_eval_checkpoints:
                    self.eval_run_tracker[f'{task}.{num_steps}.{metric_name}'].\
                        append(task_metric_mean)

                    best_function = max if metric_summary == 'max' else min

                    if best_function(self.eval_run_tracker[f'{task}.{num_steps}.{metric_name}']) \
                            == task_metric_mean:
                        mark_best_ckpt = True


        ### If specified, possibly saving out checkpoint 
        logger.info("*"*20)
        logger.info("Finished evaluator")
        

        if self.save_latest_checkpoint or self.save_eval_checkpoints:
            if not self.use_wandb:
                logger.error("Cannot save model checkpoint because use_wandb set to False")
            else:
                checkpoint = {
                    'learner_state_dict': learner.state_dict(),
                    'optimizer_state_dict': learner.optimizer.state_dict(),
                }

                torch.save(checkpoint, os.path.join(wandb.run.dir, "latest-checkpoint.pt"))
                wandb.save("latest-checkpoint.pt")

                if mark_best_ckpt:
                    logger.info(f"Saving new best model checkpoint at step: {num_task_batches}")
                    torch.save(checkpoint, os.path.join(wandb.run.dir,\
                                            f"checkpoint-{num_task_batches}.pt"))
                    wandb.save(f"checkpoint-{num_task_batches}.pt")


        logger.info("-"*30)
        logger.info("")

