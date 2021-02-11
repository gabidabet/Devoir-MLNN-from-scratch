from activation.ActivationFunction import ActivationFunction
import numpy as np


class Sigmoid(ActivationFunction):
    '''
    Class representing Sigmoid function
    '''

    def transfer(self, activation):

        return 1/(1+np.exp(-activation))

    def transfer_derivative(self, output):
        return output * (1 - output)
