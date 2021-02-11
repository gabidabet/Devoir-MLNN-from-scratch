from activation.ActivationFunction import ActivationFunction
import numpy as np


class Linear(ActivationFunction):

    def transfer(self, activation):
        '''
        Linear activation function
        '''
        return activation

    def transfer_derivative(self, output):
        '''
            always return 1.
        '''
        return np.ones(output.shape)
