''' Authors: Steven Sousa & Nicolas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/09/2024
    version: 1.0
    Description: This is the Linear Regression file for the Population History project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Preprocess the data to handle correlated features and create new features.
        3. Select features based on correlation with the target variable.
        4. Prepare the data for modeling and train a linear regression model.
        5. Evaluate the model and plot the results.'''

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import pandas as pd
import os
matplotlib.use('Agg')

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os, matplotlib
matplotlib.use('Agg')

def select_features(df, selected_indicator, threshold=0.9):
    """Selects features based on a higher correlation threshold with the target variable."""
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=np.number)

    # Check if the selected indicator is in the dataframe
    if selected_indicator not in numeric_df.columns:
        print(f"Indicator column '{selected_indicator}' not found or not numeric. Check data")
        return []
    
    # Calculate correlations with the target variable
    correlations = numeric_df.corr()[selected_indicator].drop(selected_indicator)
    selected_features = correlations[abs(correlations) > threshold].index.tolist()

    # If no features exceed the threshold, use all numeric features
    if not selected_features:
        print("No features found exceeding correlation threshold. Using all numeric features.")
        selected_features = numeric_df.columns.drop(selected_indicator).tolist()

    return selected_features

def prepare_data(df, selected_indicator, selected_features):
    """Prepares the data for the model using the selected features."""
    
    # Ensure 'Time' is included in the features
    if 'Time' not in selected_features:
        selected_features.append('Time')

    X = df[selected_features]
    y = df[selected_indicator]

    # Normalize 'Time' feature
    X['Time'] = X['Time'] - X['Time'].min()
    return X, y

def train_model(x_train, y_train):
    """Trains the linear regression model."""

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    """Evaluates the model and prints performance metrics."""

    y_pred = model.predict(x_test)
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', root_mean_squared_error(y_test, y_pred))
    print('R2 Score:', r2_score(y_test, y_pred))
    return y_pred

def plot_results(y_test, y_pred, indicator, selected_country, output_dir):
    """Plots the actual vs predicted results."""

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, label="Predicted")
    plt.plot(y_test, y_test, color='blue', label="Actual")
    plt.xlabel(f"Actual {indicator}")
    plt.ylabel(f"Predicted {indicator}")
    plt.title(f"Linear Regression Predictions for {indicator} in {selected_country}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'linear_regression.png'))

def analyze(df, selected_country, selected_indicator, output_dir):
    """Main function to run the linear regression analysis."""

    # Select features based on a higher correlation threshold with the target variable
    selected_features = select_features(df, selected_indicator, threshold=0.9)
    print("Selected Features:", selected_features)
    # Filter data for the selected country or use all countries
    df = df[df['Country Name'] == selected_country]
    # Prepare the data for modeling
    X, y = prepare_data(df, selected_indicator, selected_features)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # Train the model
    model = train_model(X_train, y_train)
    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)
    # Plot the results
    plot_results(y_test, y_pred, selected_indicator, selected_country, output_dir)