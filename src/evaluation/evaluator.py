__author__ = 'Richard Diehl Martinez'
''' Orchestration of model evaluation '''

import logging

from ..datasets import NLUDataLoader, NLU_DATASET_GENERATOR_MAPPING

logger = logging.getLogger(__name__)

class Evaluator(object): 
    def __init__(self, config): 
        """ Sets up dataset generators for each eval task provided in the config """

        eval_tasks_str = config.get("EVALUATION", "tasks", fallback="")
        if eval_tasks_str:
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
        logger.info("\nRunning evaluator\n")

        for idx, eval_task in enumerate(self.eval_tasks):
            logger.info("*"*20)
            logger.info(f"(Task {idx}) Running evaluation task: {eval_task}")

            dataset_generator = self.dataset_generators[eval_task]

            for finetune_dataset, evaluation_dataset in dataset_generator:
                finetune_language = finetune_dataset.language
                evaluation_language = evaluation_dataset.language
                logger.info("\t Finetuning on language: {finetune_language} - evaluating on language: {evaluation_language}")

                finetune_dataloader = NLUDataLoader(finetune_dataset, batch_size=self.batch_size)
                evaluation_dataloader = NLUDataLoader(evaluation_dataset, batch_size=self.batch_size)

                for batch in finetune_dataloader:
                    continue
                    # do something

                for batch in evaluation_dataloader:
                    continue

                #  compute metrics with predictions
                metrics = 0.0
                logger.info("\t metrics: {metric}")
            
            eval_task_metrics = 0.0
            logger.info("\t Metrics: {eval_task_metrics}")

        logger.info("*"*20)
        logger.info("\nFinished evaluator\n")
        logger.info("#"*30)


