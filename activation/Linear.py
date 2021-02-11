from activation.ActivationFunction import ActivationFunction
import numpy as np


class Linear(ActivationFunction):
    '''
    Class representing Linear function
    '''

    def transfer(self, activation):
        '''
        Linear activation
        '''
        return activation

    def transfer_derivative(self, output):
        '''
        Derivation of linear Function
        '''
        return np.ones(output.shape)
