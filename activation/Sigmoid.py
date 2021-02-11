from activation.ActivationFunction import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):

    def transfer(self, activation):
        '''
        Rectified Linear Unit activation function.
        '''
        return 1/(1+np.exp(-activation))

    def transfer_derivative(self, output):
        '''
        We are using the Rectified Linear Unit transfer function, the derivative of which can be calculated as follows:
        derivative = 0 for x<0 ; 1 for x>=0
        '''
        return output * (1 - output)
