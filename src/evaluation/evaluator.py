__author__ = 'Richard Diehl Martinez'
''' Orchestration of model evaluation '''

import logging

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
        """ Sets up dataset generators for each eval task provided in the config """

        eval_tasks_str = config.get("EVALUATION", "tasks", fallback="")
        if eval_tasks_str == "":
            logger.warning("Initializing evaluator with no eval tasks")

        self.eval_tasks = eval_tasks_str.split(',')

        self.dataset_generators = {task: NLU_DATASET_GENERATOR_MAPPING[task](config)
                                    for task in self.eval_tasks}

        self.batch_size = config.getint("EVALUATION", "batch_size", fallback=32)

    def run(self, learner):
        """ 
        Runs evaluation of the passed in learner on the self.eval_tasks evaluation tasks. 
        For each of the 
        """

        logger.info("#"*30)
        logger.info("Running evaluator")

        for idx, eval_task in enumerate(self.eval_tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {eval_task}")

            eval_task_params = TASK_EVALUATION_PARAMS[eval_task]
            eval_task_type = eval_task_params['task_type']

            dataset_generator = self.dataset_generators[eval_task]

            if eval_task_type == "classification": 
                finetune_method = learner.run_finetuning_classification
                inference_method = learner.run_inference_classification
            else: 
                raise Exception(f"Invalid task type: {eval_task_type} for task: {eval_task}")

            for subtask_idx, (finetune_dataset, evaluation_dataset) in enumerate(dataset_generator):
                finetune_language = finetune_dataset.language
                evaluation_language = evaluation_dataset.language
                logger.info(f"\t Finetuning on language: {finetune_language} - evaluating on language: {evaluation_language}")

                finetune_dataloader = NLUDataLoader(finetune_dataset, batch_size=self.batch_size)
                evaluation_dataloader = NLUDataLoader(evaluation_dataset, batch_size=self.batch_size)

                if not dataset_generator.use_few_shot_adaptation:
                    # we are doing zero-shot adaptation so the initial finetuning is always the same
                    if subtask_idx == 0:
                        inference_params = finetune_method(finetune_dataloader, **eval_task_params)
                else:
                    inference_params = finetune_method(finetune_dataloader, **eval_task_params)

                predictions, eval_loss = inference_method(evaluation_dataloader, **inference_params, adaptation_batch=None)

                #  compute metrics with predictions
                metrics = 0.0
                logger.info(f"\t \t Metrics: {metrics} - Eval Loss: {eval_loss}")
            
            eval_task_metrics = 0.0
            logger.info(f"\t (Task {idx}) Metrics: {eval_task_metrics}")

        logger.info("*"*20)
        logger.info("Finished evaluator")
        logger.info("#"*30)


