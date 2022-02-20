__author__ = 'Richard Diehl Martinez'
'''Base ABC Class for (meta) learners'''

import abc 

class BaseLearner(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def optimizer_step(self, set_zero_grad=False):
        """ 
        Take a global update step of the meta learner params; optionally set the 
        gradients of the meta learner gradient tape back to zero 
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_inner_loop(self, support_batch, query_batch=None, *args, **kwargs): 
        """ 
        Run an inner loop optimization step (in the context of meta learning); assumes 
        that the class contains the model that is to-be meta-learned.

        Args:
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
            * query_batch [optional]: same as support_batch, but for the data of the query set. This
                argument might be optional depending on how the learner is implemented.

        Returns: 
            * loss (torch.tensor): a tensor containing the loss that results from the inner loop 
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_evaluation(self, support_batch, query_batch=None, *args, **learner_kwargs):
        """ 
        Run evaluation of the model which is assumed to be a property of this class.

        Args: 
            * support_batch: a dictionary containing the following information for the support set
                * input_ids (torch.tensor): Input tensors of shape (N*K, max_seq_len)
                * label_ids (torch.tensor): Tensor of labels corresponding to masked out subword id
                * attention_mask (torch.tensor): Tensor indicating which tokens in input_ids are not pad tokens
            * query_batch: same as support_batch, but for the data of the query set 
        
        Returns: 
            * #TODO
        """
        raise NotImplementedError()