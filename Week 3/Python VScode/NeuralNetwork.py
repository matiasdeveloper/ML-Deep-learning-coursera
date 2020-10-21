import numpy as np
from Utilities.ColorConsole import bcolors

class NeuralNetwork:
    def __init__(self, X_train, y_train):
        # Training examples
        self.X = X_train
        self.Y = y_train
        # One hidden layer and one output layer
        self.W1 = []
        self.b1 = []
        self.W2 = []
        self.b2 = []
        # Historic parameters
        self.parameters_hist = []
        self.cost_hist = []
        self.iters_hist = []
        
    def layer_sizes(self, X, n_h, Y):
        n_x = X.shape[0]
        n_h = n_h
        n_y = Y.shape[0]

        print('Initialize neural network with one hidden layer with this sizes: ')
        print(" - The size of the input layer is: n_x = " + str(n_x))
        print(" - The size of the hidden layer is: n_h = " + str(n_h))
        print(" - The size of the output layer is: n_y = " + str(n_y))
        print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
        return (n_x, n_h, n_y)

    def random_init(self, n_x, n_h, n_y):
        # Random initialize parameters for one hidden layer
        np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

        self.W1 = np.random.randn(n_h, n_x) *0.01
        self.b1 = np.zeros(shape=(n_h, 1))
        self.W2 = np.random.randn(n_y, n_h) *0.01
        self.b2 = np.zeros(shape=(n_y, 1))

        print('Random initialize parameters:')
        print("W1 = " + str(self.W1))
        print("b1 = " + str(self.b1))
        print("W2 = " + str(self.W2))
        print("b2 = " + str(self.b2))
        print(bcolors.OKCYAN + '|-------------<>--------------|' + bcolors.ENDC)
        
        parameters = {"W1": self.W1,
                    "b1": self.b1,
                    "W2": self.W2,
                    "b2": self.b2}

        return parameters
        
    def sigmoid(self, z):
        s = 1 / (1 + np.exp(-z))
        return s

    def tanh(self,x):
        t = np.tanh(x)
        return t

    def forward(self, X):

        # Implement forward propagation to calculate A2
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.tanh(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.sigmoid(Z2)

        cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

        return A2, cache

    def compute_cost(self, A2, Y):
        m = Y.shape[1]
        
        # Compute the cross-entropry cost
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1-A2))
        cost = - np.sum(logprobs) / m
        
        # makes sure cost is the dimension we expect. 
        cost = float(np.squeeze(cost))     # E.g., turns [[17]] into 17 
        
        return cost
    
    def backpropagation(self, cache, X, Y):
        m = X.shape[1]

        A1 = cache['A1']
        A2 = cache['A2']
        
        # Implement the backward propagation 
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(self.W2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}

        return grads
    
    def update_parameters(self, grads, learning_rate):
        
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        
        # Update rule
        self.W1 = self.W1 - (learning_rate * dW1)
        self.b1 = self.b1 - (learning_rate * db1)
        self.W2 = self.W2 - (learning_rate * dW2)
        self.b2 = self.b2 - (learning_rate * db2)
        
        parameters = {"W1": self.W1,
                "b1": self.b1,
                "W2": self.W2,
                "b2": self.b2}

        return parameters

    def train_model(self, l_units,iters=10000, lr=1.2, print_cost=False):
        np.random.seed(3)
        sizes = self.layer_sizes(self.X, l_units, self.Y)
        n_x = sizes[0]
        n_y = sizes[2]
        n_h = sizes[1]
        
        # Random initialize
        parameters = self.random_init(n_x, n_h, n_y)

        #Loop gradient descent
        for i in range(0, iters):
            # Forward propagation
            A2, cache = self.forward(self.X)
            # compute cost
            cost = self.compute_cost(A2, self.Y)
            # Backward propagation
            grads = self.backpropagation(cache, self.X, self.Y)
            # Update parameters
            parameters = self.update_parameters(grads, lr)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 500 == 0:
                print (bcolors.FAIL + "Cost after iteration %i: %f" %(i, cost) + bcolors.ENDC)
                self.cost_hist.append(cost)
                self.iters_hist.append(i)
                self.parameters_hist.append(parameters)
                
        return parameters

    def predict(self, parameters, X):
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward(X)
        predictions = np.round(A2)

        return predictions
    
    def accuracy(self, predictions):
        print ('Accuracy: %d' % float((np.dot(self.Y,predictions.T) + np.dot(1- self.Y, 1-predictions.T))/float(self.Y.size)*100) + '%')