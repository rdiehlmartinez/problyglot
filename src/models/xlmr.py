__author__ = 'Richard Diehl Martinez'

import typing 
import torch
import json
import logging

from transformers import XLMRobertaModel
from .base import BaseModel
from ..utils import device

logger = logging.getLogger(__name__)

@BaseModel.register
class XLMR(XLMRobertaModel):

    @classmethod
    def from_kwargs(cls, pretrained_model_name='xlm-roberta-base', 
                         trainable_layers=None,
                         **kwargs): 
        """ Loading in huggingface XLM-R model for masked LM """
        
        if pretrained_model_name:
            model = cls.from_pretrained(pretrained_model_name).to(device)

            # https://multimodal-toolkit.readthedocs.io/en/latest/_modules/transformers/configuration_xlm_roberta.html
            if 'base' in pretrained_model_name:
                model._hidden_dim = 768
            elif 'large' in pretrained_model_name:
                model._hidden_dim = 1024
            else:
                raise Exception(f"Cannot infer hidden dimension for {cls} model: {pretrained_model_name}")

        else:
            raise NotImplementedError(f"{cls} can only be initialized from a pretrained model")

        
        # update model to require gradients only for trainable layers
        if trainable_layers: 
            if isinstance(trainable_layers, str):
                trainable_layers = json.loads(trainable_layers)
            
            for name, param in model.named_parameters():
                if any(f"layer.{layer_num}" in name for layer_num in trainable_layers):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            logger.warning("No specific layers specified to meta learn")

        return model

    @property
    def hidden_dim(self): 
        return self._hidden_dim