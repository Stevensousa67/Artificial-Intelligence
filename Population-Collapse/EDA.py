''' Authors: Steven Sousa & Nicholas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/07/2024
    Description: This is the EDA file for the Population Collapse project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Perform Exploratory Data Analysis (EDA) as defined by the NIST publication and described by John Tukey on the data.
    version: 1.0'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils

def data_summary_overview(data):
    '''Print the summary statistics and overview of the dataset.'''
    print('Data Overview:\n', data.head())
    print('Data Summary:\n', data.describe())

def missing_data_analysis(data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cmap='viridis', cbar=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Missing Data Heatmap')
    plt.show()

def correlation_analysis(data):
    '''Plot the correlation matrix heatmap for the numeric data in the dataset.'''

    numeric_data = data.drop(columns=['Country', 'Time'])
    correlation_matrix = numeric_data.corr()

    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, center=0, square=True)
    plt.title('Correlation Matrix Heatmap')
    plt.show()

def time_series_analysis(data, country, indicators):
    '''Plot the time series data for a specific country and selected indicators.'''

    for indicator in indicators:
        indicator_data = data[data['Country'] == country][['Time', indicator]]
        
        if indicator_data.empty:
            print(f"No data available for {indicator} in {country}. Skipping...")
            continue
        
        ax = plt.subplots(figsize=(14, 8))
        indicator_values = indicator_data.set_index('Time')[indicator]
        years = indicator_values.index
        
        ax.plot(years, indicator_values.values.flatten(), label=indicator)
        ax.set_title(f'{country} - {indicator}')
        ax.set_xlabel('Year')
        ax.set_ylabel(indicator)
        ax.legend()
        ax.set_xticks(years)
        ax.set_xticklabels(years, rotation=45)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

def distribution_analysis(data, country, target_indicators):
    '''Plot the distribution of target indicators for a specific country.'''

    for series_name, label in target_indicators.items():
        if series_name not in data.columns:
            print(f"{series_name} is not present in the dataset. Skipping...")
            continue
        
        indicator_data = data[data['Country'] == country][['Country', 'Time', series_name]]
        values = indicator_data[series_name].values.flatten()
        values = values[~np.isnan(values)]
        
        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True, bins=30)
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {label} for {country}')
        plt.show()

def main():
    '''Main function to perform EDA on the population data.'''
    
    population_data = utils.load_data(utils.FILE_PATH)
    if population_data is None:
        return

    sns.set(style='whitegrid')

    data_summary_overview(population_data)
    missing_data_analysis(population_data)
    correlation_analysis(population_data)

    unique_countries = population_data['Country'].unique()
    selected_country = utils.get_user_selection(unique_countries, "Available Countries:")

    indicators = population_data.columns[population_data.columns.get_loc('Time') + 1:]
    time_series_analysis(population_data, selected_country, indicators)

    target_indicators = {
        'Age dependency ratio (% of working-age population)': 'Age Dependency Ratio',
        'Age dependency ratio, young': 'Young Age Dependency Ratio',
        'Age dependency ratio, old': 'Old Age Dependency Ratio',
        'Birth rate, crude (per 1,000 people)': 'Birth Rate',
        'Death rate, crude (per 1,000 people)': 'Death Rate',
        'Fertility rate, total (births per woman)': 'Fertility Rate',
        'Life expectancy at birth, total (years)': 'Life Expectancy',
        'Net migration': 'Net Migration',
        'Population, female': 'Female Population',
        'Population, male': 'Male Population',
        'Population, total': 'Total Population',
        'Rural population': 'Rural Population',
        'Urban population': 'Urban Population',
        'Rural population (% of total population)': 'Rural Population (%)',
        'Urban population (% of total population)': 'Urban Population (%)',
    }

    distribution_analysis(population_data, selected_country, target_indicators)

if __name__ == "__main__":
    main()