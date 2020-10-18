import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, X_train, y_train):
        self.X = X_train
        self.Y = y_train

        self.layer_sizes = layer_sizes
        self.w1, self.w2, self.b1, self.b2 = self.random_init()

        self.parameters_hist = []
        self.cost_hist = []
        self.iters_hist = []
        
    
    def random_init(self):
        n_x = self.layer_sizes[0] 
        n_y = self.layer_sizes[1]
        n_h = self.layer_sizes[2]

        print("The size of the input layer is: n_x = " + str(n_x))
        print("The size of the hidden layer is: n_h = " + str(n_h))
        print("The size of the output layer is: n_y = " + str(n_y))
        print('-----------------------------')

        # Random initialize parameters for one hidden layer
        np.random.seed(2)  # we set up a seed so that your output matches ours although the initialization is random.

        w1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros(shape=(n_h, 1))
        w2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros(shape=(n_y, 1))

        print('Random initialize parameters:')
        print("W1 = " + str(w1))
        print("b1 = " + str(b1))
        print("W2 = " + str(w2))
        print("b2 = " + str(b2))
        print('-----------------------------')
        
        assert (w1.shape == (n_h, n_x))
        assert (b1.shape == (n_h, 1))
        assert (w2.shape == (n_y, n_h))
        assert (b2.shape == (n_y, 1))

        return w1, w2, b1, b2
    
    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def tanh(self,x):
        t = np.tanh(x)
        return t

    def forward(self, X):
        w1 = self.w1
        b1 = self.b1
        w2 = self.w2
        b2 = self.b2

        # Implement forward propagation to calculate A2
        Z1 = np.matmul(w1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.matmul(w2, A1) + b2
        A2 = self.sigmoid(Z2)

        cache = {"Z1": Z1,
            "A1": A1,
            "Z2": Z2,
            "A2": A2}

        return A2, cache

    def compute_cost(self, A2, Y):
        m = Y.shape[1]
        
        # Compute the cross-entropry cost
        logprobs = (1/m)*np.sum(np.multiply(np.log(A2),Y) +np.multiply(np.log(1-A2),(1-Y)))
        cost = - np.sum(logprobs)
        # makes sure cost is the dimension we expect. 
        # E.g., turns [[17]] into 17 
        cost = float(np.squeeze(cost))
        assert(isinstance(cost, float))
        
        return cost
    
    def backpropagation(self, cache, X, Y):
        m = X.shape[1]
        w1 = self.w1
        w2 = self.w2

        A1 = cache['A1']
        A2 = cache['A2']
        # Implement the backward propagation 
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.multiply(np.dot(w2.T, dZ2), 1 - np.power(A1, 2))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        grads = {"dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2}

        return grads
    
    def update_parameters(self, grads, lr):
        dW1 = grads['dW1']
        db1 = grads['db1']
        dW2 = grads['dW2']
        db2 = grads['db2']
        # Update rule
        self.w1 = self.w1 - lr * dW1
        self.b1 = self.b1 - lr * db1
        self.w2 = self.w2 - lr * dW2
        self.b2 = self.b2 - lr * db2
        
        parameters = {"W1": self.w1,
                "b1": self.b1,
                "W2": self.w2,
                "b2": self.b2}

        return parameters

    def train_model(self, iters=10000, lr=1.2, print_cost=False):
        #Loop gradient descent
        for i in range(0, iters):
            # Forward propagation
            A2, cache = self.forward(self.X)
            # compute cost
            cost = self.compute_cost(A2, self.Y)
            # Backward propagation
            grads = self.backpropagation(cache, self.X, self.Y)
            # Update parameters
            self.parameters_hist = self.update_parameters(grads, lr)
            
            # Print the cost every 1000 iterations
            if print_cost and i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, cost))
                self.cost_hist.append(cost)
                self.iters_hist.append(i)
                
        return self.parameters_hist

    def predict(self, X):
        # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
        A2, cache = self.forward(X)
        predictions = A2 > 0.5

        return predictions