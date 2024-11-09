''' Authors: Steven Sousa & Nicholas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/07/2024
    version: 1.0
    Description: This is the EDA file for the Population Collapse project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Perform Exploratory Data Analysis (EDA) as defined by the NIST publication and described by John Tukey on the data.
    Outcomes: 
        1. Utilizing the methods .head() and .describe() from pandas libray. we have gained insight as to the contents of the data and an overview of the data, such as the mean, standard deviation, min, max.
        2. It has been observed that the data set has no missing values
        3. The correlation matrix heatmap shows that there are strong positive correlations between certain features, such as the total population and the urban population.
        4. The time series analysis shows the trends of various indicators over time for a selected country.
        5. The distribution analysis provides insights into the distribution of key indicators for a specific country.
    
   EDA Steps:
        Step 0 (Data cleaning through Microsoft Excel): 
            - Downloaded raw data from World Bank DataBank (Population Estimates and Projections).
            - Link to data source: https://databank.worldbank.org/source/population-estimates-and-projections
            - Raw data initially contained all countries recognized by the UN and an extra 2 "series" called "Mortality rate, infant (per 1,000 live births)" and "Mortality rate, neonatal (per 1,000 live births)"
            - Due to a large amount of countries lacking data for "Mortality rate, infant (per 1,000 live births)" and "Mortality rate, neonatal (per 1,000 live births)", these two series were dropped from the dataset.
            - Countries lacking data on any other remaining series were also removed.
            - Total countries dropped: 8 (Andorra, Monaco, Palau, San Marino, Seychelles, Liechtenstein, Faroe Islands, Kosovo)

        Step 1 (Data Summary & Overview):
            - Purpose of this step was to get a sense of the data distribution, central tendencies, and to check for any null values.
            - During this step, each column of the spreadsheet had the following information calculated:
                - count, mean, std, min, 25%, 50%, 75%, max, null values, data type

        Step 2 (Missing Data Analysis):
            - Purpose of this step was to see through a heatmap if there were any null values left over.
            - Any null values would appear on the heatmap as a yellow line pertaining to the series on the x-axis of the heatmap.
            - Y-axis of the heatmap refers to the row number in the excel file.

        Step 3 (Correlation Analysis):
            - Purpose of this step was to see how the series correlate to each other and display the correlation on a heatmap.
            - First the data was prepared by excluding columns "Country" and "Time" to ensure only numeric columns were used for calculating correlations.
            - Generated a correlation matrix, which reveals how strongly each variable is related to others.
            - Created a heatmap to visualize the correlation matrix, ranging from -1.00 (less correlated) to 1.00 (more correlated).

        Step 4 (Time Series Analysis):
            - Purpose of this step was for the user to provide the program a Country name so that they can visualize trends from 1970 - 2022 through all the different series in the dataset.
            - Graphs built utilzing line charts, where x-axis are the years and the y-axis is the series.
            - Program will iterate over all series in the dataset and plot the data over time for the specified country

        Step 5 (Distribution Analysis):
            - Purpose of this step was to show the user's selected country series distribution utilizing histograms.'''

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

    data_summary_overview(population_data)  # EDA Step 1
    missing_data_analysis(population_data)  # EDA Step 2
    correlation_analysis(population_data)   # EDA Step 3

    unique_countries = population_data['Country'].unique()
    selected_country = utils.get_user_selection(unique_countries, "Available Countries:")

    indicators = population_data.columns[population_data.columns.get_loc('Time') + 1:]
    time_series_analysis(population_data, selected_country, indicators) # EDA Step 4

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

    distribution_analysis(population_data, selected_country, target_indicators) # EDA Step 5

if __name__ == "__main__":
    main()