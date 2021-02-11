import numpy as np
from tabulate import tabulate


class ClassificationEvaluator:
    '''
        Classificatin evaluator class
        Used to evaluate Models
    '''

    def compute_confusion_matrix(self, true, pred):
        '''
        Confuison matrix depending on dataset classes
        @input true : real values from dataset
        @pred : values predicted by the model
        it gives sometimes errors not correct 100%
        '''

        self.classes = np.unique(true)
        for i in range(0, len(pred)):
            if pred[i] >= self.classes.size:
                pred[i] = self.classes[1]
        K = len(self.classes)
        result = np.zeros((K, K))
        for i in range(len(true)):
            result[true[i]][pred[i]] += 1
        self.result = result
        return result

    def print_nicely(self):
        '''
        Print the confusion matrix nicely
        '''
        print(tabulate(self.result, headers=self.classes))

    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0
