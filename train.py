from breast_cancer_pred_mlp import NeuralNet
from data_processing import split_data
import numpy as np
import csv

models_parameters = {}
counter = 0
for i in range(20):
    X_train, X_test, Y_train, Y_test = split_data
    nn = NeuralNet(layers_dims=[X_train.shape[0], 15, 10, 5, 1])  # [X_train.shape[0], 3, 1]

    # Training function
    def train(X, Y, num_iterations, lr):
        costs = []

        for i in range(num_iterations):
            AL, caches = nn.forward(X)
            L2_loss = nn.loss(AL, Y, lambd=0.0005)
            grads = nn.backpropagation(AL, Y, caches)
            parameters = nn.grad_desc(grads, lr=lr)

            if i % 100 == 0:
                print(f'Cost on iteration {i}: {L2_loss}')
                costs.append(L2_loss)

        return parameters

    # Make predictions to calculate accuracy
    def predictions(X):
        AL, _ = nn.forward(X)

        return AL

    parameters = train(X_train, Y_train, num_iterations=8000, lr=0.02) # 0.007

    Y_train_preds = predictions(X_train)
    Y_test_preds = predictions(X_test)

    train_mse = np.mean((Y_train_preds - Y_train)) ** 2
    test_mse = np.mean((Y_test_preds - Y_test)) ** 2

    print(f'Training MSE: {train_mse:.7f}')
    print(f'Test MSE: {test_mse:.7f}')

    np.savez(f'/Users/Admin/Documents/MachineLearning/breast-cancer-predictions-scratch/models/model_parameters{counter}.npz')

    with open("models_parameters.csv", "a", newline="") as f:
        w = csv.writer(f)
        if counter == 0:
            w.writerow(["Model ID", "Train MSE", "Test MSE"])
        w.writerow([f"model_{counter}", train_mse, test_mse])

    counter += 1

