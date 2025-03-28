import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('data.csv')
print(dataset.head())

# Split data in X and Y
def data_split(dataset):
    X = dataset[["radius_mean","texture_mean","perimeter_mean","area_mean",
                "smoothness_mean","compactness_mean","concavity_mean",
                "concave points_mean","symmetry_mean","fractal_dimension_mean",
                "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
                "compactness_se","concavity_se","concave points_se","symmetry_se",
                "fractal_dimension_se","radius_worst","texture_worst","perimeter_worst",
                "area_worst","smoothness_worst","compactness_worst","concavity_worst",
                "concave points_worst","symmetry_worst","fractal_dimension_worst"]]
    Y = dataset[['diagnosis']]

    return X, Y

# Randomize samples order
def shuffle_data(X, Y):
    X = X.sample(frac=1)
    Y = Y.sample(frac=1)

    return X, Y

# Split in train and test sets
def train_val_split(X, Y):
    train_ratio = 0.8
    test_ratio = 0.2

    X_train = X[:int(len(X) * train_ratio)]
    X_test = X[:int(len(X) * test_ratio)]

    Y_train = Y[:int(len(Y) * train_ratio)]
    Y_test = Y[:int(len(Y) * test_ratio)]

    return X_train, X_test, Y_train, Y_test

# Normalization
def input_normalization():
    pass


