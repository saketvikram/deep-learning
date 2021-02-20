import numpy as np
import pandas as pd

def step(weighted_sum): # step activation function
    """
    The step activation is applied to the perceptron output that
    returns 0 if the weighted sum is less than 0 and 1 otherwise 
    """
    return (weighted_sum > 0) * 1

def sigmoid(z):
    """The sigmoid activation function on the input x"""
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, W, b):
    """
     Computes the forward propagation operation of a perceptron and 
     returns the output after applying the step activation function
    """
    weighted_sum = np.dot(X, W) + b # calculate the weighted sum of X and W
    prediction = sigmoid(weighted_sum) # apply the sigmoid activation function
    return prediction

def gradient(X, Y, Y_predicted):
    """"Gradient of weights and bias"""
    Error = Y_predicted - Y # Calculate error
    dW = np.dot(X.T, Error) # Compute derivative of error w.r.t weight, i.e., (target - output) * x
    db = np.sum(Error) # Compute derivative of error w.r.t bias
    return dW, db # return derivative of weight and bias

def update_parameters(W, b, dW, db, learning_rate):
    """Updating the weights and bias value"""
    W = W - learning_rate * dW # update weight
    b = b - learning_rate * db # update bias
    return W, b # return weight and bias


def train(X, Y, W, b):
    epochs = 10
    learning_rate = 0.1
    """Training the perceptron using batch update"""
    for i in range(epochs): # loop over the total epochs
        Y_predicted = forward_propagation(X, W, b) # compute forward pass
        dW, db = gradient(X, Y, Y_predicted) # calculate gradient
        W, b = update_parameters(W, b, dW, db, learning_rate) # update parameters

    return W, b

# Initializing values
# Data retrieval and preparation.
dataset = pd.read_csv("iris.csv", skiprows=1) # read data from csv
X = dataset.iloc[0:100, [0, 1, 2, 3]].values # features
Y = dataset.iloc[0:100, 4].values # labels
Y = np.where(Y == 'Iris-setosa', 0, 1) # if value is iris setosa, assign it 0 and 1 otherwise
learning_rate = 0.5 # learning rate
weights = np.array([0.0, 0.0, 0.0, 0.0]) # weights of perceptron
bias = 0.0 # bias value
print("Target value\n", Y)

# Model training
W, b = train(X, Y, weights, bias)


# Predicting value
A2 = forward_propagation(X, W, b)
print("Predicted value")
Y_predicted = (A2 > 0.5) * 1
print(Y_predicted)

# Comparing predicted and target outcome
comparison = Y_predicted == Y
equal_arrays = comparison.all()

print("Y == Y_predicted:", equal_arrays)