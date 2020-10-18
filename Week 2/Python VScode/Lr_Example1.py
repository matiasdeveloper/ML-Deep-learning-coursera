import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import misc  
from PIL import Image
from Lr_Utils import load_dataset
from matplotlib.pyplot import imread
from skimage.transform import rescale, resize, downscale_local_mean
from NeuralNetwork import NeuralNetwork

#General Architecture
# Init parameters
# Learn parameters
# Use learned to predict
# Analyses result

# Loading dataset
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
index = 2
plt.imshow(train_set_x_orig[index])
plt.show(block=False)
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

# Preprocesing data
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Reshape training examples and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

print("\nThe shape of train set is: " + str(train_set_x.shape))
print("The shape of test set is: " + str(test_set_x.shape) + "\n")

NeuralN = NeuralNetwork(train_set_x, train_set_y, test_set_x, test_set_y)
NeuralN.train_model(5000,0.005,True)

# Plot learning curve (with costs)
costs = np.squeeze(NeuralN.cost_hist)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(NeuralN.lr))
plt.show(block=False)

# Test with my own image
my_image = "CatTest5.jpg" 

# We preprocess the image to fit your algorithm.
fname = "images/" + my_image
image = np.array(imread(fname))
image = image/255.
my_image = resize(image, (num_px,num_px)).reshape((1, num_px*num_px*3)).T
my_predicted_image = NeuralN.predict(my_image)

plt.imshow(image)
plt.show(block=False)
print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")