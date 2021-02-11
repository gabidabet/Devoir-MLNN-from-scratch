import numpy as np
from tabulate import tabulate


class ClassificationEvaluator:

    def compute_confusion_matrix(self, true, pred):
        self.classes = np.unique(true)
        K = len(self.classes)
        result = np.zeros((K, K))
        for i in range(len(true)):
            result[true[i]][pred[i]] += 1
        self.result = result
        return result

    def print_nicely(self):
        print(tabulate(self.result, headers=self.classes))
