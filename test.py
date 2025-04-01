import numpy as np
import pandas as pd
from breast_cancer_pred_mlp import NeuralNet
from data_processing import process_data

parameters = np.load('models/model_parameters3.npz')
data = pd.read_csv('/Users/Admin/Documents/MachineLearning/breast-cancer-predictions-scratch/synthetic_test_data.csv')
_, _, _, _, scaler = process_data(dataset=data, return_scaler=True)

new_x = data.drop(columns=['id', 'diagnosis'])
new_x = scaler.transform(new_x).T

nn = NeuralNet(layers_dims=[new_x.shape[0], 15, 10, 5, 1])
nn.parameters = {key: parameters[key] for key in parameters.files}

predictions, _ = nn.forward(new_x)
predicted_class = (predictions > 0.5).astype(int)

print(f'Probabilities: {predictions.round(2)}')
for prediction in predicted_class[0]:
    if prediction == 1:
        print('Maligant')
    elif prediction == 0:
        print('Benign')
