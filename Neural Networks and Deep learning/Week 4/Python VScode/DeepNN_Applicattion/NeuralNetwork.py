import numpy as np
from dnn_app_utils_v3 import *

class NeuralNetworkApp:
    def __init__(self, train_X, test_X, train_y, test_y):

        self.train_X = train_X
        self.train_y = train_y
        self.test_X = test_X
        self.test_y = test_y

        self.cost_hist = []
        self.iters_hist = []
        self.lr = 0

    def initialize_parameters_deep(self, layer_dims):
        # Random initialize parameters for L layers
        np.random.seed(1)
        param = {}
        L = len(layer_dims)            # number of layers in the network

        for l in range(1, L):
            ### START CODE HERE ### (≈ 2 lines of code)
            param['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt (layer_dims [l-1])
            param['b' + str(l)] = np.zeros((layer_dims[l], 1))
            ### END CODE HERE ###

        return param

    def linear_forward(self, A, W, b):
        # Implement the pre activation function 
        Z = np.dot(W, A) + b
        
        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)
        
        return Z, cache

    def linear_forward_activation(self, A_prev, W, b, activation):
        # Implement the forward propagation for LINEAR>activation layer
        if activation == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        if activation == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            
        cache = (linear_cache, activation_cache)
        
        return A, cache


    def linear_forward_model(self, X, parameters):
        # Implement forward propagation for LINEAR>Relu && LINEAR>Sigmoid

        caches = []
        A = X
        L = len(parameters) // 2                  # number of layers in the neural network
        
        # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
        for l in range(1, L):
            A_prev = A 
            A, cache = self.linear_forward_activation(A_prev, parameters['W' + str(l)],
                                                parameters['b' + str(l)],
                                                activation='relu')
            caches.append(cache)
        
        # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
        ### START CODE HERE ### (≈ 2 lines of code)
        AL, cache = self.linear_forward_activation(A, parameters['W' + str(L)],
                                                parameters['b' + str(L)],
                                                activation='sigmoid')
        caches.append(cache)
    
        assert(AL.shape == (1,X.shape[1]))

        return AL, caches


    def compute_cost(self, AL, Y):
        # Implement the cost function defined by equation

        m = Y.shape[1]
        # Compute Cost Function For N-Layers
        cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1-Y, np.log(1 - AL)))
        cost = np.squeeze(cost)

        return cost
     
    def linear_backward(self, dZ, cache):
        # Implement the linear portion of backward propagation for a single layer (layer l)
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = (1/m) * np.dot(dZ, A_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def linear_activation_backward(self, dA, cache, activation):
        # Implement the backward propagation for the LINEAR->ACTIVATION layer
        linear_cache, activation_cache = cache

        if activation == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db
    
    def linear_backward_model(self, AL, Y, caches):
        # Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

        """
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
        """

        grads = {}
        L = len(caches)
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # Initializing the backpropagation
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
        
        # For last layer L SIGMOID -> LINEAR implement activation backaward sigmoid
        current_cache = caches[L - 1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activation='sigmoid')
    
        for l in reversed(range(L-1)):
            # For every layer RELU -> LINEAR implement activation backward relu
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, activation='relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    def update_parameters(self, parameters, grads, lr):
        # Update parameters for L layers
        L = len(parameters) // 2
        param = {}
        for l in range(L):
            param['W' + str(l+1)] = parameters['W' + str(l+1)] - np.multiply(lr, grads['dW' + str(l+1)])
            param["b" + str(l+1)] = parameters["b" + str(l+1)] - np.multiply(lr, grads["db" + str(l+1)])


        return param
        
    
    def L_layer_model(self, train_X, train_Y, layer_dims, lr = 0.0075, iters=3000, print_cost = False):
        ## L-Layers model for a NN

        # Step 1: Initialize randomly parameters for all layers
        np.random.seed(1)
        costs = []

        self.lr = lr
        parameters = self.initialize_parameters_deep(layer_dims)

        # Step 2: Loop (Gradient descent)
        for i in range(0, iters):
            # Forward propagation
            Al, caches = self.linear_forward_model(train_X, parameters)
            # Compute Cost
            cost = self.compute_cost(Al, train_Y)
            # Backward propagation
            grads = self.linear_backward_model(Al, train_Y, caches)
            # Update parameters
            parameters = self.update_parameters(parameters, grads, lr)

            # Print the cost every 100 training example
            if print_cost and i % 100 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
            if print_cost and i % 100 == 0:
                self.cost_hist.append(cost)
                self.iters_hist.append(i)

        self.parameters = parameters
        return parameters






    