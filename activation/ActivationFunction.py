from abc import ABCMeta, abstractmethod


class ActivationFunction:
    '''
    Abstract base class for activation function
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def transfer(self, activation):
        raise NotImplementedError()

    @abstractmethod
    def transfer_derivative(self, output):
        raise NotImplementedError()
