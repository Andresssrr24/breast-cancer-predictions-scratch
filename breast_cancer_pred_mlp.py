import numpy as np
from data_processing import process_data, dataset

X_train, X_test, Y_train, Y_test = process_data(dataset)
print(X_train.shape)

class NeuralNet:
    def __init__(self, layers_dims):
        self.parameters = {}
        L = len(layers_dims)

        # Parameters initialization
        for l in range(1, L):
            self.parameters[f'W{str(l)}'] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
            self.parameters[f'b{str(l)}'] = np.zeros((layers_dims[l], 1)) 

    # /---------------------------------------/
    # Activation functions
    def relu(self, z):
        return np.maximum(0, z)

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    # /---------------------------------------/
    # Forward propagation
    def linear_forward(self, a, W, b):
        Z = np.dot(W, a) + b
        cache = (a, W, b)

        return Z, cache

    def forward_activation(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == 'relu':
            A = self.relu(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)
        
        activation_cache = z
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def forward(self, X):
        caches = []

        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A

            AL, cache = self.forward_activation(A_prev, self.parameters[f'W{str(l)}'], 
                                               self.parameters[f'b{str(L)}'], 'relu')

            caches.append(cache)

        AL, cache = self.forward_activation(A_prev, self.parameters[f'b{str(l)}'],
                                            self.parameters[f'b{str(L)}'], 'sigmoid')
        caches.append(cache)    

        return AL, caches

    # /---------------------------------------/
    # Loss computation  
    def loss(self, AL, Y, parameters, lambd):
        pass
    # /---------------------------------------/
    # Backpropagation

    # /---------------------------------------/
    # Activation functions
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)

forward_test = NeuralNet()


