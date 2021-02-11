from activation.ActivationFunction import ActivationFunction
import numpy as np


class LeakyReLU(ActivationFunction):

    def transfer(self, activation):
        '''
        Rectified Linear Unit activation function.
        '''
        return activation * (activation > 0) + 0.01 * (activation <= 0)

    def transfer_derivative(self, output):
        '''
        We are using the Rectified Linear Unit transfer function, the derivative of which can be calculated as follows:
        derivative = 0 for x<0 ; 1 for x>=0
        '''
        return 1. * (output > 0) + 0.01 * (output <= 0)
