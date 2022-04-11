__author__ = 'Richard Diehl Martinez'
''' Orchestration of model evaluation '''

import logging
import os
import torch
import wandb
import numpy as np

from collections import defaultdict
from ..datasets import NLUDataLoader, NLU_DATASET_GENERATOR_MAPPING

logger = logging.getLogger(__name__)

TASK_EVALUATION_PARAMS = {
    "xnli": {
        "task_type": "classification",
        "n_classes": 3, 
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

        eval_tasks_str = config.get("EVALUATION", "tasks", fallback="")
        if eval_tasks_str == "":
            logger.warning("Initializing evaluator with no eval tasks")

        self.eval_tasks = eval_tasks_str.split(',')

        self.dataset_generators = {task: NLU_DATASET_GENERATOR_MAPPING[task](config)
                                    for task in self.eval_tasks}

        self.batch_size = config.getint("EVALUATION", "batch_size", fallback=32)

        # maximum number of bathes to finetune model on 
        self.max_finetuning_batch_steps = config.getint("EVALUATION", "max_finetuning_batch_steps",
                                                        fallback=-1)
        assert(self.max_finetuning_batch_steps != 0),\
            "max_finetuning_batch_steps must either be -1 or >1"

        self.save_checkpoints = config.getboolean("EVALUATION", "save_checkpoints", fallback=False)
        # possibly track of previous runs of the evaluator for checkpoint purposes
        if self.save_checkpoints:
            self.eval_run_tracker = defaultdict(list)

        self.use_wandb = config.getboolean('EXPERIMENT', 'use_wandb', fallback=True)

    @staticmethod
    def compute_accuracy(predictions, evaluation_dataloader):
        """ Computes accuracy of predictions extraction from data of the evaluation_dataloader """
        labels = []
        for data_batch in evaluation_dataloader:
            labels.extend(data_batch['label_ids'].tolist())
        
        accuracy = (np.array(predictions) == np.array(labels)).sum()/len(labels)
        return accuracy

    def run(self, learner, num_task_batches=0):
        """ 
        Runs evaluation of the passed in learner on the self.eval_tasks evaluation tasks. 
        Loops over each of the evaluation tasks in self.eval_tasks and for each eval_tasks 
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

        save_current_checkpoint = False

        for idx, eval_task in enumerate(self.eval_tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {eval_task}")

            eval_task_params = TASK_EVALUATION_PARAMS[eval_task]
            eval_task_params['max_finetuning_batch_steps'] = self.max_finetuning_batch_steps
            eval_task_type = eval_task_params['task_type']

            dataset_generator = self.dataset_generators[eval_task]

            if eval_task_type == "classification": 
                finetune_method = learner.run_finetuning_classification
                inference_method = learner.run_inference_classification
                compute_metric = self.compute_accuracy
                metric_name = "acc"
                metric_summary = 'max'
            else: 
                logger.exception(f"Invalid task type: {eval_task_type} for task: {eval_task}")
                raise Exception(f"Invalid task type: {eval_task_type} for task: {eval_task}")

            eval_task_metrics = []
            eval_task_losses = []

            for subtask_idx, (finetune_dataset, evaluation_dataset) in enumerate(dataset_generator):
                finetune_lng = finetune_dataset.language
                evaluation_lng = evaluation_dataset.language
                logger.info(f"\t Finetuning on: {finetune_lng} - evaluating on: {evaluation_lng}")

                finetune_dataloader = NLUDataLoader(finetune_dataset,
                                                    batch_size=self.batch_size)
                evaluation_dataloader = NLUDataLoader(evaluation_dataset,
                                                      batch_size=self.batch_size)

                # Calling on finetuning method which returns a set of trained parameters that 
                # can be used for inference (inference_params)
                if not dataset_generator.use_few_shot_adaptation:
                    # we are doing zero-shot adaptation --> initial finetuning is always the same
                    if subtask_idx == 0:
                        inference_params = finetune_method(finetune_dataloader, **eval_task_params)
                else:
                    inference_params = finetune_method(finetune_dataloader, **eval_task_params)

                adaptation_batch = None
                if dataset_generator.adapt_on_eval:
                    if type(learner).__name__ != "Platipus":
                        logger.warning("(ignoring adapt_on_eval) - learner is not 'platipus'")
                    else:    
                        # adapt on the first batch of the evaluation datalaoder
                        adaptation_batch = next(iter(evaluation_dataloader))

                predictions, eval_loss = inference_method(evaluation_dataloader,
                                                          adaptation_batch=adaptation_batch,
                                                          **inference_params)

                # For logging of metric
                if self.use_wandb:
                    wandb.define_metric(f"{eval_task}.{evaluation_lng}.{metric_name}",
                                        step_metric="num_task_batches", summary=metric_summary)
                    wandb.define_metric(f"{eval_task}.{evaluation_lng}.loss",
                                        step_metric="num_task_batches", summary='min')

                # compute metrics using predictions 
                metric = compute_metric(predictions, evaluation_dataloader)
                logger.info(f"\t \t {metric_name}: {metric:.4f} - Eval Loss: {eval_loss:.4f}")
                if self.use_wandb:
                    wandb.log({eval_task: {
                                    evaluation_lng: {
                                        "loss": eval_loss,
                                        metric_name: metric,
                                    },
                                },
                            "num_task_batches": num_task_batches
                            })
            
                eval_task_metrics.append(metric)
                eval_task_losses.append(eval_loss)
                
            eval_task_metrics_mean = sum(eval_task_metrics)/len(eval_task_metrics)
            eval_task_loss_mean = sum(eval_task_losses)/len(eval_task_losses)

            if self.use_wandb:
                wandb.define_metric(f"{eval_task}.{metric_name}", step_metric="num_task_batches",
                                    summary=metric_summary)
                wandb.define_metric(f"{eval_task}.loss", step_metric="num_task_batches",
                                    summary='min')

                wandb.log({eval_task: {
                            "loss": eval_task_loss_mean,
                            metric_name: eval_task_metrics_mean,
                        },
                        "num_task_batches": num_task_batches
                        })

            logger.info(f"\t (Task {idx}) Avg. {metric_name}: {eval_task_metrics_mean:.4f}")
            logger.info(f"\t (Task {idx}) Avg Loss: {eval_task_loss_mean:.4f}")

            # Possibly writing out checkpoint 
            if self.save_checkpoints:
                self.eval_run_tracker[f'{eval_task}.{metric_name}'].append(eval_task_metrics_mean)

                best_function = max if metric_summary == 'max' else min

                if best_function(self.eval_run_tracker[f'{eval_task}.{metric_name}']) \
                        == eval_task_metrics_mean:
                    save_current_checkpoint = True


        logger.info("*"*20)
        logger.info("Finished evaluator")
        if save_current_checkpoint:
            if not self.use_wandb:
                logger.error("Cannot save model checkpoint because use_wandb set to False")
            else:
                logger.info(f"Saving model checkpoint at task batch number: {num_task_batches}")
                checkpoint = {
                    'learner_state_dict': learner.state_dict(),
                    'optimizer_state_dict': learner.optimizer.state_dict(),
                }
                torch.save(checkpoint, os.path.join(wandb.run.dir,
                                                    f"checkpoint-{num_task_batches}.pt"))
                wandb.save(f"checkpoint-{num_task_batches}.pt")
        logger.info("-"*30)
        logger.info("")

