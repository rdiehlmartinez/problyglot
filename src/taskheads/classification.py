__author__ = 'Richard Diehl Martinez'
""" Defines a task head for classification tasks """

import math 
import torch
import torch.nn as nn
import torch.nn.functional as F 

from .base import TaskHead

class ClassificationHead(TaskHead):
    """ Task head for classification tasks"""

    loss_function = torch.nn.CrossEntropyLoss()

    def __call__(self, model_output, labels, weights):
        """ 
        Runs a forward pass of the classification head. Architecture inspired by the huggingface
        implementation of RobertaLMHead 
        """

        if "fc_weight" in weights:
            fc_weights = {"weight": weights["fc_weight"], "bias": weights["fc_bias"]}

            model_output = F.linear(model_output, **fc_weights)
            model_output = F.gelu(model_output)
            model_output = F.layer_norm(model_output, (model_output.size(-1),))

        classifier_weights = {"weight": weights["classifier_weight"],
                              "bias": weights["classifier_bias"]}

        logits = F.linear(model_output, **classifier_weights)
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
        * task_head_weights (nn.ParameterDict): {
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
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

    task_head_weights = nn.ParameterDict({
        "classifier_weight": nn.Parameter(classifier_weight),
        "classifier_bias": nn.Parameter(classifier_bias)
    })


    return task_head_weights

@TaskHead.register_initialization_method
def classification_random_fc(base_model_hidden_dim, n_classes, **kwargs):
    """
    Initializes classification task head using a random Xavier-He initialization method. 
    Unlike the classification_random initialization method, this method also includes a 
    fully connected layer that is inserted before the final classification output layer. 

    Args: 
        * base_model_hidden_dim (int): The hidden dimensions of the outputs of the base_model 
        * n_classes (int): Number of classes to classify over 
    Returns: 
        * task_head_weights (nn.ParameterDict): {
            * fc_weight -> (nn.Parameter): weight matrix of fully connected layer
            * fc_bias -> (nn.Parameter): bias vector of fully connected layer 
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
            }
    """
    # Little bit of a hack - can initialize weights of FC layer by
    # repurposing classification_random
    fc_head_weights = classification_random(base_model_hidden_dim, base_model_hidden_dim, **kwargs)
    classifier_weights = classification_random(base_model_hidden_dim, n_classes, **kwargs)

    task_head_weights = nn.ParameterDict({
        "fc_weight": fc_head_weights["classifier_weight"],
        "fc_bias": fc_head_weights["classifier_bias"],
        "classifier_weight": classifier_weights["classifier_weight"],
        "classifier_bias": classifier_weights["classifier_bias"]
    })

    return task_head_weights


@TaskHead.register_initialization_method
def classification_protomaml(base_model_hidden_dim, n_classes, functional_model, params,
                             data_batch, device, **kwargs):
    """
    Initializes task head using the protomaml (prototypical network + MAML) method. 

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
        * task_head_weights (nn.ParameterDict): {
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
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

    task_head_weights = nn.ParameterDict({
        "classifier_weight": nn.Parameter(classifier_weight),
        "classifier_bias": nn.Parameter(classifier_bias)
    })

    return task_head_weights


@TaskHead.register_initialization_method
def classification_protomaml_fc(base_model_hidden_dim, n_classes, **kwargs):
    """
    Same as protomaml, expect also adds a fully connected layer (FC) of dimension 
    base_model_hidden_dim. This FC connected layer is initialized randomly. 

    Args: 
        * base_model_hidden_dim (int): The hidden dimensions of the outputs of the base_model 
        * n_classes (int): Number of classes to classify over 

        * kwargs must contain the following args: 
            * functional_model (higher.MonkeyPatched): The 'functionalized' version of the 
                base model
            * params ([torch.Tensor]): List of tensor weights storing the model weights.
            * data_batch (dict): Batch of data for a forward pass through the model 
                (see run_inner_loop for information on the data structure).
            * device (str): Device type ('cuda' or 'cpu')
    Returns: 
        * task_head_weights (nn.ParameterDict): {
            * fc_weight -> (nn.Parameter): weight matrix of fully connected layer
            * fc_bias -> (nn.Parameter): bias vector of fully connected layer 
            * classifier_weight -> (nn.Parameter): classification weight matrix
            * classifier_bias -> (nn.Parameter): classification bias vector
            }
    """

    # Little bit of a hack - can initialize weights of FC layer by
    # repurposing classification_random
    fc_head_weights = classification_random(base_model_hidden_dim, base_model_hidden_dim, **kwargs)

    protomaml_weights = classification_protomaml(base_model_hidden_dim, n_classes, **kwargs)

    task_head_weights = nn.ParameterDict({
        "fc_weight": fc_head_weights["classifier_weight"],
        "fc_bias": fc_head_weights["classifier_bias"],
        "classifier_weight": protomaml_weights["classifier_weight"],
        "classifier_bias": protomaml_weights["classifier_bias"]
    })

    return task_head_weights
