import pandas as pd
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
dataset = dataset.drop(columns='Unnamed: 32')

def process_data(dataset, target_col='diagnosis', test_size=0.2, random_state=42):
    # Split data in X and Y
    X = dataset.drop(columns=[target_col, 'id'])
    Y = pd.get_dummies(dataset[target_col], drop_first=True).astype(int)

    # Shuffle data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)

    #Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).T  # (30, 455)
    X_test = scaler.fit_transform(X_test).T  # (30, 114)

    # Reshape Y
    Y_train = Y_train.values.reshape(1, -1)  # (1, 455)
    Y_test = Y_test.values.reshape(1, -1)  # (1, 114)

    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = process_data(dataset)
