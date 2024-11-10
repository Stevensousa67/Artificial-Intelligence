''' Authors: Steven Sousa & Nicholas Pace
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

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import utils

def preprocess_data(df, correlation_threshold=0.8):
    """Preprocesses the dataframe to handle correlated features and create proportions."""
    
    # Create new features based on existing ones
    df['Population, total'] = df['Population, female'] + df['Population, male']
    df['Young Dependency Ratio'] = df['Age dependency ratio, young']
    df['Old Dependency Ratio'] = df['Age dependency ratio, old']
    df['Urban Population (%)'] = df['Urban population (% of total population)']
    df['Rural Population (%)'] = df['Rural population (% of total population)']
    
    # Drop redundant columns
    columns_to_drop = [
        'Population, female', 'Population, male', 'Age dependency ratio', 
        'Age dependency ratio (% of working-age population)', 'Rural population',
        'Urban population', 'Rural population (% of total population)',
        'Population ages 00-04, female','Population ages 00-04, male',
        'Population ages 05-09, female','Population ages 05-09, male',
        'Population ages 10-14, female','Population ages 10-14, male',
        'Population ages 15-19, female','Population ages 15-19, male',
        'Population ages 20-24, female','Population ages 20-24, male',
        'Population ages 25-29, female','Population ages 25-29, male',
        'Population ages 30-34, female','Population ages 30-34, male',
        'Population ages 35-39, female','Population ages 35-39, male',
        'Population ages 40-44, female','Population ages 40-44, male',
        'Population ages 45-49, female','Population ages 45-49, male',
        'Population ages 50-54, female','Population ages 50-54, male',
        'Population ages 55-59, female','Population ages 55-59, male',
        'Population ages 60-64, female','Population ages 60-64, male',
        'Population ages 65-69, female','Population ages 65-69, male',
        'Population ages 70-74, female','Population ages 70-74, male',
        'Population ages 75-79, female','Population ages 75-79, male',
        'Population ages 80 and above, female','Population ages 80 and above, male'
    ]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove highly correlated features
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
    df = df.drop(columns=to_drop, errors='ignore')

    return df

def select_features(df, selected_indicator, threshold=0.3):
    """Selects features based on correlation with the target variable."""
    
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

def plot_results(y_test, y_pred, indicator):
    """Plots the actual vs predicted results."""

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, label="Predicted")
    plt.plot(y_test, y_test, color='blue', label="Actual")
    plt.xlabel(f"Actual {indicator}")
    plt.ylabel(f"Predicted {indicator}")
    plt.title(f"Linear Regression Predictions for {indicator}")
    plt.legend()
    plt.show()

def main():
    """Main function to run the linear regression analysis."""

    # Load the data
    population_data = utils.load_data(utils.FILE_PATH)
    if population_data is None:
        return

    # Preprocess the data
    population_data = preprocess_data(population_data)

    # Select the indicator to predict
    indicators = population_data.columns.drop(['Country', 'Time'])
    selected_indicator = utils.get_user_selection(indicators, "Select indicator to predict:")

    # Select the country or all countries
    unique_countries = population_data['Country'].unique()
    selected_country = utils.get_user_selection(unique_countries, "Select country (or type 'All' for all countries):")

    # Select features based on correlation with the target variable
    selected_features = select_features(population_data, selected_indicator, threshold=0.3)
    print("Selected Features:", selected_features)

    # Filter data for the selected country or use all countries
    if selected_country.lower() != 'all':
        population_data = population_data[population_data['Country'] == selected_country]

    # Prepare the data for modeling
    X, y = prepare_data(population_data, selected_indicator, selected_features)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    y_pred = evaluate_model(model, X_test, y_test)

    # Plot the results
    plot_results(y_test, y_pred, selected_indicator)

if __name__ == "__main__":
    main()