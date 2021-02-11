import numpy as np
import matplotlib.pyplot as plt

from MultiLayerNN import MultiLayerNN
from activation.ReLU import ReLU

from mlxtend.plotting import plot_decision_regions
# truth table of the problem : (x,y)
# x > 0 , y > 0 , xor
# T     , T     , F
# T     , F     , T
# F     , T     , T
# F     , F     , F


# this function take values normale distribution N(0,1) donc les valeurs entre -4,4 (see graphe of N(0,1))

rng = np.random.RandomState(0)
X = rng.randn(300, 2)
y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)


# Creating model
model1 = MultiLayerNN(2, 6, 1, bias_iniatlisation_methode='ones')
model2 = MultiLayerNN(2, 6, 1, bias_iniatlisation_methode='ones')
model3 = MultiLayerNN(2, 6, 1, bias_iniatlisation_methode='ones')

# Learning
print("model {} | learning_rate : {:.4f} | function : {}".format(1, 0.01, 'Relu'))
model1.fit(X, y, nbr_iter=10000, learning_rate=0.01,
           activation_function=ReLU())
print("\n\nmodel {} | learning_rate : {:.4f} | function : {}".format(
    2, 0.001, 'Relu'))
model2.fit(X, y, nbr_iter=10000, learning_rate=0.001,
           activation_function=ReLU())

print("\n\nmodel {} | learning_rate : {:.4f} | function : {}".format(
    3, 0.0001, 'Relu'))
model3.fit(X, y, nbr_iter=10000, learning_rate=0.0001,
           activation_function=ReLU())


# display code
fig = plt.figure(figsize=(10, 8))
fig = plot_decision_regions(X=X, y=y, clf=model2, legend=2)
plt.title("Relu function")
plt.show()

test_X = rng.randn(300, 2)
test_y = np.array(np.logical_xor(
    test_X[:, 0] > 0, test_X[:, 1] > 0), dtype=int)

# prediction
predicted_y = model2.predict(test_X)

# evaluation
model2.evaluate(test_y, predicted_y)

# confiusion matrix
model2.print_confusion_matrix()
