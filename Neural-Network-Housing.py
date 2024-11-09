# This assingment will predict Boston housing using Neural Networks

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
    
if __name__ == '__main__':
    # Load the California Housing dataset and create DataFrame
    california_housing = fetch_california_housing()
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    Y = pd.Series(california_housing.target)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42)

    # Create Neural Network - i/p = 64, hidden = 32, o/p = 1
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit model
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1)

    # Predict
    y_pred = model.predict(X_test)

    # MSE
    print(f'MSE: {model.evaluate(X_test, y_test)}')
