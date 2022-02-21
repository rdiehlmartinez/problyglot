__author__ = 'Richard Diehl Martinez'
'''Base ABC class interface for base models '''

import abc 

class BaseModel(metaclass=abc.ABCMeta):

    @classmethod
    @abc.abstractmethod
    def from_kwargs(**kwargs):
        """ 
        Base models should be initialized via this class method that reads in 
        keywords from an experiment config
        """
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def hidden_dim(self):
        """ Return the hidden dimension size of the model """
        raise NotImplementedError()

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        """ Move model onto a given device """

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """ Must implement a forward pass through the model """
        raise NotImplementedError()

    @abc.abstractmethod
    def __str__(self): 
        """ For debugging purposes printing model should return something useful """
        raise NotImplementedError()