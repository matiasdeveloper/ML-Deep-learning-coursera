import numpy as np
import matplotlib.pyplot as plt
from Utilities.testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Utilities.planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets
from NeuralNetwork import NeuralNetwork

# General arquitecture for nn one hidden layer
# Random init
# Execute gradient descent (n iters)
    # Forward propagation
    # Compute cost
    # Backward propagation
    # Update parameters
# Predict graphics
print('Neural network with one hidden layer and -n- units')
print('-----------------------------')
print('-----------------------------')
print('Developed by Matias Vallejos with the course NN deepleraning.ai')
print('No linear problem')
print('-----------------------------')

input("Press Enter to continue...")

print('-----------------------------')
print('Plotting data...')
print('-----------------------------')

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

input("Press enter to get data from dataset")
print('-----------------------------')
# Sizes
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))
print('-----------------------------')

def layer_sizes(X, Y, u):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    n_h = u
    return (n_x, n_y, n_h)

input("Press enter to training model..")
print('-----------------------------')
NeuralN = NeuralNetwork(layer_sizes(X, Y, 4), X, Y)
parameters = NeuralN.train_model(10000, 1.2, True)
print('-----------------------------')

print('Press enter to visualize learn parameters..')
print('-----------------------------')
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print('-----------------------------')

print('Press enter to plot learn parameters')
print('-----------------------------')

"""
plot_decision_boundary(lambda x: NeuralN.predict(NeuralN.X_train.T), X, y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show() """
