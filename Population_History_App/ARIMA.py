''' Authors: Steven Sousa & Nicolas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/08/2024
    version: 1.0
    Description: This is the ARIMA file for the Population Collapse project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Perform decomposition on user selected country and indicator.
        3. Fit an ARIMA model to the data, make predictions & forecast, and evaluate the model.'''

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import pmdarima as pm
import utils

def plot_decomposition(decomposition, indicator):
    '''Plot time series decomposition into trend, seasonal, and residual components.'''

    fig, axes = plt.subplots(4, 1, figsize=(10, 8))
    components = ['Observed', 'Trend', 'Seasonal', 'Residual']
    data = [indicator, decomposition.trend, decomposition.seasonal, decomposition.resid]
    
    for ax, comp, d in zip(axes, components, data):
        ax.plot(d, label=comp)
        ax.set_title(comp)
        ax.legend(loc='best')
    
    plt.tight_layout()
    plt.show()

def fit_autoarima_model(train_data, test_data, forecast_periods):
    '''Fit an ARIMA model using auto_arima to automatically select the best parameters, make predictions on the test data, and forecast future values.'''

    model = pm.auto_arima(train_data, seasonal=False, stepwise=True, trace=True)
    print(f"Best ARIMA model order: {model.order}")

    # Predict on the test data
    predictions = model.predict(n_periods=len(test_data))
    rmse = root_mean_squared_error(test_data, predictions)
    print(f'RMSE: {rmse}')
    
    # Forecast future values
    forecast = model.predict(n_periods=forecast_periods)
    
    return predictions, forecast

def plot_forecast(train_data, test_data, predictions, forecast, country, indicator, forecast_periods):
    '''Plot the actual test data, the predicted values from the ARIMA model, and the forecasted future values.'''

    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Train Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, predictions, label='Predictions')
    
    # Start the forecast in 2023 and extend it for the specified forecast period
    forecast_start = test_data.index[-1] + pd.DateOffset(years=1)
    forecast_index = pd.date_range(start=forecast_start, periods=forecast_periods, freq='YS')
    forecast_series = pd.Series(forecast.values, index=forecast_index)
    
    # Plotting forecast with the corrected index
    plt.plot(forecast_series.index, forecast_series, label='Forecast')
    plt.legend(loc='best')
    plt.title(f'ARIMA Model Predictions and Forecast for {indicator} in {country}')
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.show()

def main():
    '''Main function for ARIMA modeling and forecasting.'''
    population_data = utils.load_data(utils.FILE_PATH)
    if population_data is None:
        return

    selected_country = utils.get_user_selection(population_data['Country'].unique(), "Select country:")
    selected_indicator = utils.get_user_selection(population_data.columns.drop(['Country', 'Time']), "Select indicator:")

    country_data = population_data[population_data['Country'] == selected_country]
    time_series = country_data[['Time', selected_indicator]].set_index('Time')
    time_series.index = pd.to_datetime(time_series.index, format='%Y').to_period('Y').to_timestamp()

    # Plot decomposition if data is sufficient
    if len(time_series) >= 2:
        decomposition = seasonal_decompose(time_series, model='additive', period=1)
        plot_decomposition(decomposition, time_series[selected_indicator])
    
    train_data = time_series[selected_indicator].iloc[:int(len(time_series) * 0.8)]
    test_data = time_series[selected_indicator].iloc[int(len(time_series) * 0.8):]

    forecast_periods = 10
    predictions, forecast = fit_autoarima_model(train_data, test_data, forecast_periods)
    plot_forecast(train_data, test_data, predictions, forecast, selected_country, selected_indicator, forecast_periods)

if __name__ == "__main__":
    main()