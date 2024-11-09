import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import root_mean_squared_error
import pandas as pd
import pmdarima as pm
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

def fit_autoarima_model(train_data, test_data):
    '''Fit an ARIMA model using auto_arima to automatically select the best parameters, and make predictions on the test data.'''

    model = pm.auto_arima(train_data, seasonal=True, m=12, stepwise=True, trace=True)
    print(f"Best ARIMA model order: {model.order}")

    predictions = model.predict(n_periods=len(test_data))
    rmse = root_mean_squared_error(test_data, predictions)
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
    
    # Convert Time to datetime and set as index (required for ARIMA modeling)
    population_data['Time'] = pd.to_datetime(population_data['Time'], format='%Y')
    population_data = population_data.set_index(['Country', 'Time'])
    population_data = population_data[~population_data.index.duplicated(keep='first')]

    # Check unique countries and get the user's selection
    unique_countries = population_data.index.get_level_values('Country').unique()
    selected_country = utils.get_user_selection(unique_countries, "Available Countries:")

    # Filter data for the selected country and get available indicators
    country_data = population_data[population_data.index.get_level_values('Country') == selected_country]
    indicators = country_data.columns[1:]  # Exclude 'Time' and 'Country' columns
    selected_indicator = utils.get_user_selection(indicators, "\nAvailable Indicators:")
    indicator_data = country_data[selected_indicator]
    indicator_data = indicator_data.loc[selected_country]

    # Decompose the time series
    decomposition = seasonal_decompose(indicator_data, model='additive', period=12)
    plot_decomposition(decomposition, indicator_data)

    # Split the data into training and test sets (50% split)
    split_index = len(indicator_data) // 2
    train_data = indicator_data[:split_index]
    test_data = indicator_data[split_index:]

    # Fit the ARIMA model using auto_arima
    predictions = fit_autoarima_model(train_data, test_data)
    plot_predictions(test_data, predictions, selected_country, selected_indicator)

if __name__ == "__main__":
    main()