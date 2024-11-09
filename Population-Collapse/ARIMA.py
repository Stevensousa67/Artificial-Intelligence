''' Authors: Steven Sousa & Nicholas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/08/2024
    version: 1.0
    Description: This is the ARIMA file for the Population Collapse project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Perform seasonal decomposition on user selected country and indicator.
        3. Fit an ARIMA model to the data, make predictions, and evaluate the model.'''

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import utils

def plot_decomposition(decomposition, indicator):
    '''Plot the decomposition of the time series data into trend, seasonal, and residual components.'''

    plt.figure(figsize=(10, 6))
    plt.subplot(411)
    plt.plot(indicator, label="Observed")
    plt.legend(loc='best')
    plt.title("Observed")

    plt.subplot(412)
    plt.plot(decomposition.trend, label='Trend')
    plt.legend(loc='best')
    plt.title('Trend')

    plt.subplot(413)
    plt.plot(decomposition.seasonal, label='Seasonal')
    plt.legend(loc='best')
    plt.title('Seasonal')

    plt.subplot(414)
    plt.plot(decomposition.resid, label='Residuals')
    plt.legend(loc='best')
    plt.title('Residuals')

    plt.tight_layout()
    plt.show()

def fit_arima_model(train_data, test_data):
    '''Fit an ARIMA model to the training data and make predictions on the test data.'''

    model = ARIMA(train_data, order=(5, 1, 0))  # (p, d, q) - Experiment with these orders
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train_data), end=len(train_data) + len(test_data) - 1)
    rmse = mean_squared_error(test_data, predictions, squared=False)
    print(f'RMSE: {rmse}')
    return predictions

def plot_predictions(test_data, predictions, country, indicator):
    '''Plot the actual test data and the predicted values from the ARIMA model.'''

    plt.figure(figsize=(10, 6))
    plt.plot(test_data, label='Actual')
    plt.plot(predictions, label='Predictions')
    plt.legend(loc='best')
    plt.title(f'ARIMA Model Predictions for {indicator} in {country}')
    plt.show()

def main():
    '''Main function to perform ARIMA modeling on the population data.'''
    
    population_data = utils.load_data(utils.FILE_PATH)
    if population_data is None:
        return

    unique_countries = population_data['Country'].unique()
    selected_country = utils.get_user_selection(unique_countries, "Available Countries:")

    country_data = population_data[population_data["Country"] == selected_country]
    indicators = country_data.columns[2:]  # Exclude 'Time' and 'Country'
    selected_indicator = utils.get_user_selection(indicators, "\nAvailable Indicators:")

    indicator_data = country_data[selected_indicator]
    decomposition = seasonal_decompose(indicator_data, model='additive', period=12)
    plot_decomposition(decomposition, indicator_data)

    split_index = len(indicator_data) // 2
    train_data = indicator_data[:split_index]
    test_data = indicator_data[split_index:]
    predictions = fit_arima_model(train_data, test_data)
    plot_predictions(test_data, predictions, selected_country, selected_indicator)

if __name__ == "__main__":
    main()