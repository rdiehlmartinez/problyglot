__author__ = 'Richard Diehl Martinez'
""" Defines a task head for classification tasks """

import math 
import torch
import torch.nn.functional as F 

from .base import TaskHead

class ClassificationHead(TaskHead):
    """ Task head for classification tasks"""

    loss_function = torch.nn.CrossEntropyLoss()

    def __call__(self, model_output, labels, weights):
        """ 
        Runs a forward pass of the classification head 
        """
        logits = F.linear(model_output, **weights)
        loss = self.loss_function(input=logits, target=labels)
        
        return (logits, loss)

@TaskHead.register_initialization_method
def classification_random(base_model_hidden_dim, n_classes, device, **kwargs):
    """
    Initializes classification task head using a random Xavier-He initialization method.

    Args: 
        * base_model_hidden_dim (int): The hidden dimensions of the outputs of the base_model 
        * n_classes (int): Number of classes to classify over 
        * device (str): Device type ('cuda' or 'cpu')
    Returns: 
        * task_head_weights (dict): {
            * weight -> (torch.Tensor): classification weight matrix
            * bias -> (torch.Tensor): classification bias vector
            }
    """
    # Xavier normal weight implementation
    std_weight = math.sqrt(2.0 / float(base_model_hidden_dim + n_classes))
    std_bias = math.sqrt(2.0 / float(n_classes))

    # weights need to be shape (out_features, in_features) to be compatible with linear layer
    classifier_weight = torch.randn((n_classes, base_model_hidden_dim), device=device) \
                            * std_weight
    classifier_bias = torch.randn((n_classes), device=device) * std_bias

    classifier_weight.requires_grad = True
    classifier_bias.requires_grad = True

    task_head_weights = {
        "weight": classifier_weight,
        "bias": classifier_bias
    }

    return task_head_weights


# Registering new task head initialization method 
@TaskHead.register_initialization_method
def classification_protomaml(base_model_hidden_dim, n_classes, functional_model, params,
                             data_batch, device, **kwargs):
    """
    Args: 
        * base_model_hidden_dim (int): The hidden dimensions of the outputs of the base_model 
        * n_classes (int): Number of classes to classify over 
        * functional_model (higher.MonkeyPatched): The 'functionalized' version of the 
            base model
        * params ([torch.Tensor]): List of tensor weights storing the model weights.
        * data_batch (dict): Batch of data for a forward pass through the model 
            (see run_inner_loop for information on the data structure).
        * device (str): Device type ('cuda' or 'cpu')
    Returns: 
        * task_head_weights (dict): {
            * weight -> (torch.Tensor): classification weight matrix
            * bias -> (torch.Tensor): classification bias vector
            }
    """

    outputs = functional_model.forward(input_ids=data_batch['input_ids'],
                                        attention_mask=data_batch['attention_mask'],
                                        params=[p for p in params])

    # outputs has form (batch_size, sequence_length, hidden_size);
    batch_size = outputs.size(0)
    last_hidden_state = outputs[torch.arange(batch_size), data_batch['input_target_idx']]

    prototypes = torch.zeros((n_classes, base_model_hidden_dim), device=device)

    for c in range(n_classes):
        idx = torch.nonzero(data_batch['label_ids'] == c).squeeze()
        if idx.nelement() != 0:
            prototypes[c] = torch.mean(last_hidden_state[idx], dim=0)
        else:
            logger.warning("ProtoMaml weight initialization missing at least one class")

    classifier_weight = 2 * prototypes
    classifier_bias = -torch.norm(prototypes, dim=1)**2

    task_head_weights = { 
        "weight": classifier_weight,
        "bias": classifier_bias
    }

    return task_head_weights