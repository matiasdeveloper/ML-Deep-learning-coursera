from DeepNeuralNetwork import DeepNN
from testCases_v4a import *

NN = DeepNN([], [], [], [])

print('')
print('Test Random Initialize For N-Layers')
parameters = NN.random_init_nl([5, 4, 1])

# Test random deep neural network initialize for n Layers
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

# Test linear pre activation forward
print('')
print('Test pre-linear Activation')

A, W, b = linear_forward_test_case()

Z, linear_cache = NN.linear_forward(A, W, b)
print("Z = " + str(Z))

# Test linear activation forward
print('')
print('Test Linear Activation Forward')

A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = NN.linear_activation_forward(A_prev, W, b, activationName = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = NN.linear_activation_forward(A_prev, W, b, activationName = "relu")
print("With ReLU: A = " + str(A))

# Test Cost Function
print('')
print('Test Cost Function')

Y, AL = compute_cost_test_case()
print("cost = " + str(NN.compute_cost(AL, Y)))

# Test pre-linear backward
print('')
print('Test PreLinear Backward')
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = NN.linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

# Test linear activation backward
print('')
print('Test Linear Activation Backward')
dAL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = NN.linear_activation_backward(dAL, linear_activation_cache, activationName = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = NN.linear_activation_backward(dAL, linear_activation_cache, activationName = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))

# Test linear model backward
print('')
print('Test Linear Model Backward')

AL, Y_assess, caches = L_model_backward_test_case()
grads = NN.L_model_backward(AL, Y_assess, caches)
print_grads(grads)

# Test updating parameters 
print('')
print('Test Update Parameters Gradient Descent')

parameters, grads = update_parameters_test_case()
parameters = NN.Update_Parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))