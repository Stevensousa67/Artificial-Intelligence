import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import utils

def preprocess_data(df):
    """Preprocesses the dataframe to handle correlated features and create proportions."""
    df['Population, total'] = df['Population, female'] + df['Population, male']
    df['Working-Age ratio'] = df['Age dependency ratio (% of working-age population)']
    df['Young Dependency Ratio'] = df['Age dependency ratio, young'] #/ df['Population, total']
    df['Old Dependency Ratio'] = df['Age dependency ratio, old'] #/ df['Population, total']
    df['Urban Population (%)'] = df['Urban population (% of total population)']
    df['Rural Population (%)'] = df['Rural population (% of total population)']
    # ... (Add other feature engineering or transformations as needed)

    # Drop redundant columns (adjust as per your analysis)
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
    return df


def select_features(df, selected_indicator, threshold=0.3):
    """Selects features based on correlation with the target variable."""

    numeric_df = df.select_dtypes(include=np.number)  # Select only numeric columns

    if selected_indicator not in numeric_df.columns:
        print(f"Indicator column '{selected_indicator}' not found or not numeric. Check data")
        return []
    
    correlations = numeric_df.corr()[selected_indicator].drop(selected_indicator)
    selected_features = correlations[abs(correlations) > threshold].index.tolist()

    if not selected_features:
        print("No features found exceeding correlation threshold. Using all numeric features.")
        selected_features = numeric_df.columns.drop(selected_indicator).tolist()

    return selected_features

def prepare_data(df, selected_indicator, selected_features):
    """Prepares the data for the model using the selected features."""
    X = df[selected_features]
    y = df[selected_indicator]

    X['Time'] = X['Time'] - X['Time'].min()
    return X, y

def split_data(x, y):
    return train_test_split(x, y, test_size=0.2, random_state=0)

def train_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
    print('R2 Score:', r2_score(y_test, y_pred))
    return y_pred

def plot_results(y_test, y_pred, indicator):  # Changed parameter to indicator
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, label="Predicted")
    plt.plot(y_test, y_test, color='blue', label="Actual")
    plt.xlabel(f"Actual {indicator}")  # Dynamic x-axis label
    plt.ylabel(f"Predicted {indicator}")  # Dynamic y-axis label
    plt.title(f"Linear Regression Predictions for {indicator}")  # Dynamic title
    plt.legend()
    plt.show()

def main():
    population_data = utils.load_data(utils.FILE_PATH)
    if population_data is None:
        return

    population_data = preprocess_data(population_data)

    indicators = population_data.columns.drop(['Country', 'Time'])
    selected_indicator = utils.get_user_selection(indicators, "Select indicator to predict:")

    unique_countries = population_data['Country'].unique()
    selected_country = utils.get_user_selection(unique_countries, "Select country (or type 'All' for all countries):")

    selected_features = select_features(population_data, selected_indicator, threshold=0.3)
    print("Selected Features:", selected_features)


    if selected_country.lower() != 'all':
        population_data = population_data[population_data['Country'] == selected_country]  # Filter data for selected country

        X, y = prepare_data(population_data, selected_indicator, selected_features)

        X_train, X_test, y_train, y_test = split_data(X, y)
    else:  # If all countries
        population_data = pd.get_dummies(population_data, columns=['Country'])  # One-hot encode countries

        X, y = prepare_data(population_data, selected_indicator, selected_features)

        X_train, X_test, y_train, y_test = split_data(X, y)

    model = train_model(X_train, y_train)
    y_pred = evaluate_model(model, X_test, y_test)
    plot_results(y_test, y_pred, selected_indicator)  # Pass selected_indicator for plot labels


if __name__ == "__main__":
    main()