import numpy as np

from activation.ReLU import ReLU
from activation.Sigmoid import Sigmoid
from activation.LeakyReLU import LeakyReLU

from evaluation.ClassificationEvaluator import ClassificationEvaluator


class MultiLayerNN:
    # constructeur 2,6,1
    def __init__(self, input, hidden, output, weights_iniatlisation_methode='random', bias_iniatlisation_methode='random'):
        self.evaluator = ClassificationEvaluator()
        np.random.seed(0)

        # weights inialization
        if weights_iniatlisation_methode == 'random':
            self.hidden_weights = np.random.rand(
                input, hidden)
        # Bias initalization
        if bias_iniatlisation_methode == 'random':
            self.hidden_bias = np.random.rand(1, hidden)
        elif bias_iniatlisation_methode == 'zeros':
            self.hidden_bias = np.zeros((1, hidden))
        elif bias_iniatlisation_methode == 'ones':
            self.hidden_bias = np.zeros((1, hidden))

        self.output_weights = np.random.rand(
            hidden, output)  # hidden are rows, output columns

        self.output_bias = np.random.rand(1, output)

    def fit(self, X, y, nbr_iter=1000, learning_rate=0.01, activation_function=ReLU()):
        self.activation_function = activation_function

        yy = y
        y = y.reshape(-1, 1)  # [[0][1]...]

        m = (nbr_iter/10)
        for i in range(nbr_iter):  # using back propagation algorithme
            hidden_layer_sum = np.dot(
                X, self.hidden_weights) + self.hidden_bias
            # every item in dataset we calculated the sum
            # hidden_layer_sum.shape gives (300,6)
            hidden_layer_out = self.activation_function.transfer(
                hidden_layer_sum)
            output_layer_sum = np.dot(
                hidden_layer_out, self.output_weights) + self.output_bias
            predicted_y = self.activation_function.transfer(output_layer_sum)

            dloss_dbias_out = (predicted_y-y) * \
                self.activation_function.transfer_derivative(predicted_y)
            dloss_dw_out = hidden_layer_out.T.dot(dloss_dbias_out)

            dloss_dbias_h = dloss_dbias_out.dot(
                self.output_weights.T)*self.activation_function.transfer_derivative(hidden_layer_out)
            dloss_dw_h = X.T.dot(dloss_dbias_h)

            self.hidden_weights -= dloss_dw_h*learning_rate
            self.hidden_bias -= np.sum(dloss_dbias_h,
                                       axis=0, keepdims=True)*learning_rate

            self.output_weights -= dloss_dw_out*learning_rate
            self.output_bias -= np.sum(dloss_dbias_out,
                                       axis=0, keepdims=True)*learning_rate

            if (i % m == 0):
                print("iteration ({}) -> loss : {:.4f} | accuracy : {:.4f} ".format(i,
                                                                                    np.mean((predicted_y-yy)**2), (yy == self.predict(X)).mean()))

    def predict(self, X):
        hidden_layer_sum = np.dot(X, self.hidden_weights) + self.hidden_bias

        hidden_layer_out = self.activation_function.transfer(hidden_layer_sum)
        output_layer_sum = np.dot(
            hidden_layer_out, self.output_weights) + self.output_bias

        predicted_y = self.activation_function.transfer(output_layer_sum)
        return np.round(predicted_y).reshape(1, -1)[0].astype(int)

    def evaluate(self, real, predicted):
        self.evaluator.compute_confusion_matrix(real, predicted)

    def print_confusion_matrix(self):
        self.evaluator.print_nicely()
