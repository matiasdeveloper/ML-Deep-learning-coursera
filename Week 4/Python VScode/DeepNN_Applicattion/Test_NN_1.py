import numpy as np
import h5py 
import matplotlib.pyplot as plt
from PIL import Image
import scipy
from scipy import ndimage   
from dnn_app_utils_v3 import *
from NeuralNetwork import NeuralNetworkApp

"""
This is the test of the neural network application

# The steps for the application are:
--------------------------------------
1. Build a model and Initialize NN
2. Initialize data
3. Initialize NN
4. Random initilize parameters
5. Implement model
    Loop gradient descent
        - Forward propagation
        - Backward propagation
        - Update parameters
9. Visualize Results
-------------------------------------
"""
# Step 1:
""" Construct the L-NeuralNetwork Model in 
NeuralNetwork.py"""

# Step 2:
# ------------------------------------
train_x_orig, train_y, test_x_orig, test_y, classes = load_data()

print("\nInitialize datasets")
print("--------------------------------------")
index = 111
plt.imshow(train_x_orig[index])
plt.show()
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print("\nNumber of training set are: ", m_train)
print("Number of testing set are: ", m_test)
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")

# Reshape the examples
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten/255
test_x = test_x_flatten/255

print ("\ntrain_X shape: " + str(train_x.shape))
print ("test_X shape: " + str(test_x.shape))

# ---------------------------------------

# Step 3:
AppNN = NeuralNetworkApp(train_x, test_x, train_y, test_y)
print("\nDeep Neural Network Model Construct in the Deep Learning Specialization by Coursera.org & deeplearning.ai")
print("--------------------------------------")
print("Random initialize parameters\n")

layers_dims = [12288, 20, 7, 5, 1] #  4-layer model

print("Number total of layers: " + str(len(layers_dims)-1))

print("\n--------------------------------------")
print("Training neural network model for L: " + str(len(layers_dims) - 1) + " ..\n")

AppNN.L_layer_model(layers_dims, iters = 2500, print_cost = True)

print("\n--------------------------------------")
print("Plotting Cost Chart\n")
# plot the cost
plt.plot(np.squeeze(AppNN.cost_hist))
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(AppNN.lr))
plt.show()

print("\n--------------------------------------")
print("Predict with learning parameters\n")
pred_train = predict(train_x, train_y, AppNN.parameters)
pred_test = predict(test_x, test_y, AppNN.parameters)


