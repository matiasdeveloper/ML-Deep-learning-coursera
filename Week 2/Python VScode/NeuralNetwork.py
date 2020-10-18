import numpy as np
#Steps
#1- Define Model (Logisitic Regression with Neural Network)
#2- Initialize Parameters
#3- Loop and learn
#   - Calculate loss (Forward Propagation)
#   - Calculate gradient (Backward Propagation)
#   - Update parameters (Gradien descent)

class NeuralNetwork:
    def __init__(self, X_train, Y_train, X_test, Y_test):
        # Init parameters
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        # Init model
        self.w, self.b = self.initialze_with_zeros(X_train.shape[0])

        self.cost_hist = []
        self.iters_hist = []

    # Sigmoid activation function
    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    # Initialize Parameters
    def initialze_with_zeros(self, dim):
        w = np.zeros((dim,1))   # Vector shape (dim,1)
        b = 0                   # Bias term
        print ("w = " + str(w))
        print ("b = " + str(b))
        return w,b
    
    # Forward and Backpropagation algorimt
    def propagate(self, X, y):
        m = X.shape[1]
        
        # Forward (From X to Cost)
        A = self.sigmoid(np.dot(self.w.T, X) + self.b)                            # compute activation 
        cost = -1/m * np.sum(y * np.log(A) + ((1-y)* np.log(1-A))) # compute cost 
        
        # Backpropagation (Find Gradient Descent)
        dw = 1/m * (np.matmul(X,(A-y).T))
        db = 1/m * (np.sum(A-y))

        assert(dw.shape == self.w.shape)
        assert(db.dtype == float)
        cost = np.squeeze(cost)
        assert(cost.shape == ())

        grads = {"dw": dw, "db":db}
        return grads, cost      

    # Optimization
    def optimize(self, X, y, print_cost = False):
        
        costs_error = []
         
        for i in range(self.iters):
            # Implement the forward and backward propagation
            grads, cost = self.propagate(X,y)
            # Get derivates from gradient descent
            dw = grads["dw"]
            db = grads["db"]
            # Update parameters
            self.w = self.w - (self.lr * dw)
            self.b = self.b - (self.lr * db)
            # Record costs
            if i % 100 == 0:
                costs_error.append(cost)
                self.cost_hist.append(cost)
                self.iters_hist.append(self.iters)
            # Print the cost every 100 training iters
            if print_cost and i % 100 == 0:
               print ("Cost after iteration %i: %f" %(i, cost))
            # Return params and grads
            params = {"w": self.w, "b": self.b}
            grads = {"dw": dw, "db": db}

        return params, grads, costs_error     

    # Train model 
    def train_model(self, iters = 2000, lr = 0.005, print_cost = True):
        # Init var
        self.iters = iters
        self.lr = lr
        # Execute gradient descent loop
        params, grads, costs = self.optimize(self.X_train, self.Y_train, print_cost)
        # Predict test/train set examples
        self.Y_prediction_test = self.predict(self.X_test)
        self.Y_prediction_train= self.predict(self.X_train)
        # Print train/test Errors
        print("\ntrain accuracy: {} %".format(100 - np.mean(np.abs(self.Y_prediction_train - self.Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(self.Y_prediction_test - self.Y_test)) * 100))
         
    # Predict if y = 1 or y = 0
    def predict(self, X):
        # Init parameters
        m = X.shape[1]
        Y_prediction = np.zeros((1,m))    
        w = self.w.reshape(X.shape[0], 1)
            
        # Compute vector "A" predicting the probabilities if is or no in the picture
        A = self.sigmoid(np.dot(w.T, X) + self.b)

        for i in range(A.shape[1]):
            # Convert probabilities A[0,i] to actual predictions p[0,i]
            if A[0,i] < 0.5:
                Y_prediction[0,i] = 0
            else: 
                Y_prediction[0,i] = 1
        
        assert(Y_prediction.shape == (1, m))
        
        return Y_prediction    
            

      



    