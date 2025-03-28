import pandas as pd
import numpy as np
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
print(dataset.head())

def process_data(dataset, target_col='diagnosis', test_size=0.2, random_state=42):
    # Split data in X and Y
    X = dataset.drop(columns=[target_col])
    Y = dataset[target_col]

    # Shuffle data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)

    #Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).T  # (32, 455)
    X_test = scaler.fit_transform(X_test).T  # (32, 114)

    # Reshape Y
    Y_train = Y_train.values.reshape(1, -1)  # (1, 455)
    Y_test = Y_test.values.reshape(1, -1)  # (1, 114)

    return X_train, X_test, Y_train, Y_test
