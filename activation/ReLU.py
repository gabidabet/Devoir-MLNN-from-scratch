from activation.ActivationFunction import ActivationFunction


class ReLU(ActivationFunction):
    '''
    Class representing ReLU function
    '''

    def transfer(self, activation):

        return activation * (activation > 0)

    def transfer_derivative(self, output):

        return 1. * (output > 0)  # stack overflow
