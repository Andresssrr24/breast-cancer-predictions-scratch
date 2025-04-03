import numpy as np
import pandas as pd
from breast_cancer_pred_mlp import NeuralNet
from data_processing import process_data

parameters = np.load('models/model_parameters76.npz')  # models/model_parameters22.npz & #12(23/30 over synthetic_test_data.csv)
data = pd.read_csv('/Users/Admin/Documents/MachineLearning/breast-cancer-predictions-scratch/synthetic_test_data.csv')
_, _, _, _, scaler = process_data(dataset=data, return_scaler=True)

new_x = data.drop(columns=['id', 'diagnosis'])
new_x = scaler.transform(new_x).T

nn = NeuralNet(layers_dims=[new_x.shape[0]])  # 10, 5, 3, 1
nn.parameters = {key: parameters[key] for key in parameters.files}

predictions, _ = nn.forward(new_x)
predicted_class = (predictions > 0.5).astype(int)
real_label = data['diagnosis']

print(f'Probabilities: {predictions.round(2)}')
counter = 2
right_preds = 0 
for prediction in predicted_class[0]:
    print(f'\nReal label: {real_label[counter-2]}')
    if prediction == 1:
        print(f'{counter}. Malignant')
        if real_label[counter-2] == 'M':
            right_preds += 1
    elif prediction == 0:
        print(f'{counter}. Benign')
        if real_label[counter-2] == 'B':
            right_preds += 1
    counter += 1

print(f'Right predictions: {right_preds}/{len(real_label)}. {right_preds*100/len(real_label):.2f}%')