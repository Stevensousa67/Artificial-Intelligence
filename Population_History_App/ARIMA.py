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

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import root_mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_acf_pacf(data, lags=None):
    '''Plot ACF and PACF dynamically based on dataset size.'''
    if lags is None:
        lags = min(20, len(data) // 2)  # Adjust based on data length
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plot_acf(data, lags=lags, ax=axes[0])
    plot_pacf(data, lags=lags, ax=axes[1])
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    plt.tight_layout()
    plt.show()

def select_arima_params(train_data, max_p=10, max_d=10, max_q=10):
    '''Iterate over p, d, q combinations to find the best ARIMA model based on AIC.'''
    best_aic = float('inf')
    best_order = None
    best_model = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(train_data, order=(p, d, q))
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_model = fit
                except Exception as e:
                    # Skip invalid combinations
                    continue

    print(f"Best ARIMA model order: {best_order} with AIC: {best_aic}")
    return best_model, best_order

def fit_manual_arima_model(train_data, test_data, forecast_periods):
    '''Fit an ARIMA model manually by selecting the best parameters, making predictions, and forecasting.'''
    plot_acf_pacf(train_data)  # Optional: Visualize ACF and PACF for guidance

    # Select ARIMA parameters based on training data
    max_p = min(len(train_data) // 3, 5)  # Adjust maximum lags based on dataset size
    max_d = 2  # Generally sufficient for most datasets
    max_q = min(len(train_data) // 3, 5)
    best_model, best_order = select_arima_params(train_data, max_p=max_p, max_d=max_d, max_q=max_q)

    # Predict on test data
    predictions = best_model.forecast(steps=len(test_data))
    rmse = root_mean_squared_error(test_data, predictions)
    print(f'RMSE on test data: {rmse}')

    # Forecast future values
    forecast = best_model.forecast(steps=forecast_periods)
    return predictions, forecast, best_order

def plot_forecast(train_data, test_data, predictions, forecast, country, indicator, forecast_periods, output_dir):
    '''Plot the actual test data, the predicted values, and forecasted future values.'''
    plt.figure(figsize=(10, 6))
    plt.plot(train_data.index, train_data, label='Train Data')
    plt.plot(test_data.index, test_data, label='Test Data')
    plt.plot(test_data.index, predictions, label='Predictions')

    # Forecast future values
    forecast_start = test_data.index[-1] + pd.DateOffset(years=1)
    forecast_index = pd.date_range(start=forecast_start, periods=forecast_periods, freq='YS')
    plt.plot(forecast_index, forecast, label='Forecast')

    plt.legend(loc='best')
    plt.title(f'ARIMA Model Predictions and Forecast for {indicator} in {country}')
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.savefig(os.path.join(output_dir, 'ARIMA_results.png'))
    plt.close()

def plot_decomposition(decomposition, time_series, output_dir):
    '''Plot the decomposition results (trend, seasonal, residual).'''
    plt.figure(figsize=(12, 8))

    # Plot the trend component
    plt.subplot(411)
    plt.plot(decomposition.trend, label='Trend')
    plt.title('Trend Component')
    plt.legend()

    # Plot the seasonal component
    plt.subplot(412)
    plt.plot(decomposition.seasonal, label='Seasonal', color='orange')
    plt.title('Seasonal Component')
    plt.legend()

    # Plot the residual component
    plt.subplot(413)
    plt.plot(decomposition.resid, label='Residual', color='green')
    plt.title('Residual Component')
    plt.legend()

    # Plot the original time series
    plt.subplot(414)
    plt.plot(time_series, label='Original', color='black')
    plt.title('Original Time Series')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ARIMA_decomposition_results.png'))
    plt.close()

def analyze(df, selected_country, selected_indicator, output_dir):
    '''Main function for ARIMA modeling and forecasting.'''
    # Filter data for the selected country and indicator
    country_data = df[df['Country Name'] == selected_country]
    time_series = country_data[['Time', selected_indicator]].set_index('Time')
    time_series.index = pd.to_datetime(time_series.index, format='%Y').to_period('Y').to_timestamp()

    # Decompose time series (only if there are enough data points)
    if len(time_series) >= 2:
        try:
            decomposition = seasonal_decompose(time_series[selected_indicator], model='additive', period=1)
            plot_decomposition(decomposition, time_series[selected_indicator], output_dir)
        except ValueError as e:
            print(f"Decomposition failed: {e}")
    else:
        print("Insufficient data for decomposition.")

    # Split data into 80/20 for training and testing
    train_data = time_series[selected_indicator].iloc[:int(len(time_series) * 0.8)]
    test_data = time_series[selected_indicator].iloc[int(len(time_series) * 0.8):]

    # Set forecast period (e.g., 10 years)
    forecast_periods = 10
    predictions, forecast, best_order = fit_manual_arima_model(train_data, test_data, forecast_periods)
    print(f"Final Model Order: {best_order}")
    
    # Plot forecast results
    plot_forecast(train_data, test_data, predictions, forecast, selected_country, selected_indicator, forecast_periods, output_dir)