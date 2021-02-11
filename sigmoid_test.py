import numpy as np
import matplotlib.pyplot as plt

from MultiLayerNN import MultiLayerNN
from activation.Sigmoid import Sigmoid

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
hidden_layer_size = 6
model1 = MultiLayerNN(2, hidden_layer_size, 1,
                      bias_iniatlisation_methode='ones')
model2 = MultiLayerNN(2, hidden_layer_size, 1,
                      bias_iniatlisation_methode='ones')
model3 = MultiLayerNN(2, hidden_layer_size, 1,
                      bias_iniatlisation_methode='ones')

# Learning
print("model {} | learning_rate : {:.4f} | function : {}".format(1, 0.01, 'sigmoid'))
model1.fit(X, y, nbr_iter=10000, learning_rate=0.01,
           activation_function=Sigmoid())

print("\n\nmodel {} | learning_rate : {:.4f} | function : {}".format(
    2, 0.001, 'sigmoid'))
model2.fit(X, y, nbr_iter=10000, learning_rate=0.001,
           activation_function=Sigmoid())

print("\n\nmodel {} | learning_rate : {:.4f} | function : {}".format(
    3, 0.0001, 'sigmoid'))
model3.fit(X, y, nbr_iter=10000, learning_rate=0.0001,
           activation_function=Sigmoid())


# display code
fig = plt.figure(figsize=(10, 8))
fig = plot_decision_regions(X=X, y=y, clf=model1, legend=2)
plt.title("Sigmoid function with {} hidden layers and learning rate {:.4}".format(
    hidden_layer_size, 0.01))
plt.show()

test_X = rng.randn(300, 2)
test_y = np.array(np.logical_xor(
    test_X[:, 0] > 0, test_X[:, 1] > 0), dtype=int)

# prediction
predicted_y = model1.predict(test_X)

# evaluation
model1.evaluate(test_y, predicted_y)

# confiusion matrix
model1.print_confusion_matrix()

print("accuracy {:.4f}".format(model1.get_accuracy()))
