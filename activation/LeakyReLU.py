from activation.ActivationFunction import ActivationFunction
import numpy as np


class LeakyReLU(ActivationFunction):
    '''
    Class representing LeakyRelu function
    '''

    def transfer(self, activation):
        '''
        LeakyReLU
        '''
        return activation * (activation > 0) + 0.01 * (activation <= 0)

    def transfer_derivative(self, output):
        '''
        LeakyReLU derivative
        '''
        return 1. * (output > 0) + 0.01 * (output <= 0)
