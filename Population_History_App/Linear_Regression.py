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

def reshape_data(x_train, x_test):
    """Reshapes the data for the model."""
    
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)
    return x_train, x_test

def train_model(x_train, y_train):
    """Trains the linear regression model."""

    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test, output_dir):
    """Evaluates the model and saves performance metrics to an Excel file."""

    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create a DataFrame with the metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error', 'Mean Squared Error', 'R2 Score'],
        'Value': [mae, mse, r2]
    })

    # Save the DataFrame to an Excel file
    metrics_df.to_excel(os.path.join(output_dir, 'linear_regression_metrics.xlsx'), index=False)
    return y_pred

def plot_results(y_test, y_pred, indicator, selected_country, output_dir):
    """Plots the actual vs predicted results and saves the plot as an image."""

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, label="Predicted")
    plt.plot(y_test, y_test, color='blue', label="Actual")
    plt.xlabel(f"Actual {indicator}")
    plt.ylabel(f"Predicted {indicator}")
    plt.title(f"Linear Regression Predictions for {indicator} in {selected_country}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'linear_regression_results.png'))
    plt.close()

def analyze(data, indicator, country, output_dir):
    """Starting point for the linear regression analysis. Receives the dataset, target indicator, country, and output directory from the main script."""

    sns.set(style='whitegrid')
    X_train, X_test, y_train, y_test = train_test_split(data['Time'], data[indicator], test_size=0.2, random_state=0)
    reshape_data(X_train, X_test)
    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test, output_dir)
    plot_results(y_test, y_pred, indicator, country, output_dir)