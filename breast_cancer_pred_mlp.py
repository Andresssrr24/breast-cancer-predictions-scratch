import numpy as np
from data_processing import process_data, dataset

X_train, X_test, Y_train, Y_test = process_data(dataset)
print(X_train.shape)

class NeuralNet:
    def __init__(self, layer_dims):
        self.weights = []
        self.biases = []

    # /---------------------------------------/
    # Activation functions
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)

    # /---------------------------------------/
    # Forward propagation
    def forward_propagation(self, Z):
        pass


