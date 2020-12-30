import numpy as np
import matplotlib.pyplot as plt
from Utilities.testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from Utilities.planar_utils import plot_decision_boundary, load_planar_dataset, load_extra_datasets
from NeuralNetwork import NeuralNetwork
from Utilities.ColorConsole import bcolors

# General arquitecture for nn one hidden layer
# Random init
# Execute gradient descent (n iters)
    # Forward propagation
    # Compute cost
    # Backward propagation
    # Update parameters
# Predict graphics

print('')
print(bcolors.BOLD + bcolors.UNDERLINE + 'Neural network with one hidden layer and -n- units' + bcolors.ENDC)
print(bcolors.OKGREEN + '>> Developed by Matias Vallejos <<' + bcolors.ENDC)
print('Course deeplearning.ai (Bibliography)')

print('')
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
input(bcolors.WARNING +'Press enter to plot data..' + bcolors.ENDC)
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)

print('View data in the graph')
print('The X and Y axis in spectral mode.')
X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
plt.show()

print('')
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
input(bcolors.WARNING +'Press enter to get data..' + bcolors.ENDC)
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)

# Sizes
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]  # training set size

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

print('')
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
input(bcolors.WARNING + 'Press enter to training model..' + bcolors.ENDC)
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)

# Training the neural network with one hidden layer
NeuralN = NeuralNetwork(X, Y)
parameters = NeuralN.train_model(4, 10000, 1.12, True)

print('')
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
print(bcolors.WARNING + 'Press enter to visualize learned parameters..' + bcolors.ENDC)
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)

# Print the learned parameters
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

print('')
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
input(bcolors.WARNING + 'Press enter to visualize training accuracy' + bcolors.ENDC)
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)

predictions = NeuralN.predict(parameters, X)
NeuralN.accuracy(predictions)

print('')
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
input(bcolors.WARNING + 'Press enter to plot learn parameters' + bcolors.ENDC)
print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)

print('View data in the graph')
print('Visualize the parameters learned with the neural network')
print('The neural resolved a one problem of non linear random data')

# Plot the leraned parameters in a graphics
plot_decision_boundary(lambda x: NeuralN.predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
