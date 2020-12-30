import numpy as np
from DeepNeuralNetwork_utils_v2 import *

class DeepNN:
    def __init__(self, X_Train, y_Train, X_Test, y_Test):
        #Initialize NN Datasets
        self.X = X_Train
        self.Y = y_Train
        #Random Initialize Parameters
        self.parameters = []
    
    def random_init_nl(self, layers_dims):
        """
        Arguments:
        layer_dims -- python array (list) containing the dimensions of each layer in our network
        
        Returns:
        parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
        """
        np.random.seed(3)
        parameters = {}
        L = len(layers_dims)

        for l in range(L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))

        self.parameters = parameters

        return parameters
    
    def linear_forward(self, A, W, b):
        # Implement the pre activation function 
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache
    
    def linear_activation_forward(self, A_prev, W, b, activationName):
        # Implement the forward propagation for LINEAR>activation layer
        if activationName == "sigmoid":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = sigmoid(Z)
        
        elif activationName == "relu":
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A, activation_cache = relu(Z)
            
        cache = (linear_cache, activation_cache)
        
        return A, cache
    
    def L_model_forward(self, X, parameters):
        # Implement forward propagation for LINEAR>Relu && LINEAR>Sigmoid
        caches = []
        A = X
        L = len(parameters) // 2

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)],
                                            parameters['b' + str(l)],
                                            activationName='relu')
            caches.append(cache)
        
        Al, cache = self.linear_activation_forward(A, parameters['W' + str(l)],
                                            parameters['b' + str(l)],
                                            activationName='sigmoid')
        caches.append(cache)

        return Al, caches

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
    
    def linear_activation_backward(self, dA, cache, activationName):
        # Implement the backward propagation for the LINEAR->ACTIVATION layer
        linear_cache, activation_cache = cache

        if activationName == "relu":
            dZ = relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activationName == "sigmoid":
            dZ = sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        return dA_prev, dW, db
    
    def L_model_backward(self, AL, Y, caches):
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
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache, activationName='sigmoid')
    
        for l in reversed(range(L-1)):
            # For every layer RELU -> LINEAR implement activation backward relu
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l+1)], current_cache, activationName='relu')
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    def Update_Parameters(self, parameters, grads, lr):
        # Update parameters using gradient descent
        L = len(parameters) // 2

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - np.multiply(lr, grads["dW" + str(l+1)])
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - np.multiply(lr, grads["db" + str(l+1)])

        return parameters
    
        

    