import h5py 
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy import ndimage
from dnn_app_utils_v3 import load_data
from NeuralNetwork import NeuralNetworkApp
from testCases_v4a import *

"""
Testing NN

"""
# Test random initialize parameters
NN = NeuralNetworkApp([],[],[],[])

print('\nTEST RANDOM INIT')
parameters = NN.initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# Test linear activation forward
print('\nTEST FORWARD LINEAR ACTIVATION')

# Test linear forward
A, W, b = linear_forward_test_case()

Z, linear_cache = NN.linear_forward(A, W, b)
print("Z = " + str(Z))

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = NN.linear_forward_activation(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = NN.linear_forward_activation(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))

X, parameters = L_model_forward_test_case_2hidden()
AL, caches = NN.linear_forward_model(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))

# Test cost function
print('\nTEST COST FUNCTION')
Y, AL = compute_cost_test_case()

print("cost = " + str(NN.compute_cost(AL, Y)))

# Test linear backward
print('\nTEST BACKWARD LINEAR ACTIVATION')
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = NN.linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = NN.linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = NN.linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

AL, Y_assess, caches = L_model_backward_test_case()
grads = NN.linear_backward_model(AL, Y_assess, caches)
print_grads(grads)

# Test update parameters
print('\nTEST UPDATE PARAMETERS')
parameters, grads = update_parameters_test_case()
parameters = NN.update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))