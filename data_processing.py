import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('data.csv')
dataset = dataset.drop(columns='Unnamed: 32')

def process_data(dataset, target_col='diagnosis', test_size=0.2, random_state=42, return_scaler=False):
    # Split data in X and Y
    X = dataset.drop(columns=[target_col, 'id'])
    Y = dataset[target_col].map({'M':1, 'B':0}).values

    # Shuffle data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state, shuffle=True)

    #Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.fit_transform(X_test) 

    # Reshape X and Y
    X_train, X_test = X_train.T, X_test.T  # (30, 455), (30, 114)
    Y_train, Y_test = Y_train.reshape(1, -1), Y_test.reshape(1, -1)  # (1, 455), (1, 114)

    return (X_train, X_test, Y_train, Y_test, scaler) if return_scaler else (X_train, X_test, Y_train, Y_test)

X_train, X_test, Y_train, Y_test = process_data(dataset)

_, X_new, _, _, scaler = process_data(dataset, return_scaler=True)
