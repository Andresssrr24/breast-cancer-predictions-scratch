import numpy as np

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
    def linear_forward(self, A, W, b):
        Z = np.dot(W, A) + b
        cache = (A, W, b)

        return Z, cache

    def forward_activation(self, A_prev, W, b, activation):
        Z, linear_cache = self.linear_forward(A_prev, W, b)

        if activation == 'relu':
            A = self.relu(Z)
        elif activation == 'sigmoid':
            A = self.sigmoid(Z)

        activation_cache = Z
        cache = (linear_cache, activation_cache)

        return A, cache
    
    def forward(self, X):
        caches = []

        A = X
        L = len(self.parameters) // 2

        for l in range(1, L):
            A_prev = A

            A, cache = self.forward_activation(A_prev, 
                                               self.parameters[f'W{str(l)}'], 
                                               self.parameters[f'b{str(l)}'], 
                                               'relu')
            caches.append(cache)

        AL, cache = self.forward_activation(A, 
                                            self.parameters[f'W{str(L)}'],
                                            self.parameters[f'b{str(L)}'], 
                                            'sigmoid')
        caches.append(cache)    

        return AL, caches

    # /---------------------------------------/
    # Loss computation  
    def loss(self, AL, Y, lambd):
        m = Y.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10)  # Avoid log 0

        cost = -(1/m) * np.sum(Y*np.log(AL) + (1-Y) * np.log(1-AL))

        L2 = 0
        for key in self.parameters.keys():
            if 'W' in key:
                L2 += np.sum(np.square(self.parameters[key]))
        L2 = (lambd/2) * L2

        cost_L2 = cost + L2
        cost_L2 = np.squeeze(cost_L2)

        return cost_L2
                             
    # /---------------------------------------/
    # Backpropagation
    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]

        dW = 1/m * np.dot(dZ, A_prev.T)
        db = 1/m * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, db
    
    def backward_activation(self, dA, cache, activation):
        linear_cache, activation_cache = cache
        Z = activation_cache

        if activation == 'relu':
            dZ = dA * (np.where(Z > 0, 1, 0))  # relu derivative
        elif activation == 'sigmoid':
            s = self.sigmoid(Z)
            dZ = dA * (s * (1 - s))

        dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db
    
    def backpropagation(self, AL, Y, caches):
        grads = {}

        L = len(caches)
        m = AL.shape[1]
        AL = np.clip(AL, 1e-10, 1 - 1e-10)  # Avoid Y/0
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

        current_cache = caches[L-1]
        dA_prev_temp, dW_temp, db_temp = self.backward_activation(dAL, current_cache, 'sigmoid')
        grads['dA' + str(L-1)] = dA_prev_temp
        grads['dW' + str(L)] = dW_temp
        grads['db' + str(L)] = db_temp

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.backward_activation(grads['dA' + str(l+1)], current_cache, 'relu')
            grads['dA' + str(l)] = dA_prev_temp
            grads['dW' + str(l+1)] = dW_temp
            grads['db' + str(l+1)] = db_temp

        return grads
    
    # /---------------------------------------/
    # Optimization
    def grad_desc(self, grads, lr):
        L = len(self.parameters) // 2

        for l in range(L):
            self.parameters['W' + str(l+1)] = self.parameters['W' + str(l+1)] - lr * grads['dW' + str(l+1)]
            self.parameters['b' + str(l+1)] = self.parameters['b' + str(l+1)] - lr * grads['db' + str(l+1)]

        return self.parameters