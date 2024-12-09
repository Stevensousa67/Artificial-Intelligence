''' Authors: Steven Sousa & Nicolas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/07/2024
    version: 1.0
    Description: This is the EDA file for the Population History project. This file has the following objectives:
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
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import os
matplotlib.use('Agg')

def data_summary_overview(data, output_dir):
    '''Print the summary statistics and overview of the dataset and save the plot.'''

    print('Data Overview:\n', data.head())
    print('Data Summary:\n', data.describe())
    # Save the data overview and summary as Excel files
    overview_path = os.path.join(output_dir, 'data_overview.xlsx')
    summary_path = os.path.join(output_dir, 'data_summary.xlsx')
    with pd.ExcelWriter(overview_path) as writer:
        data.head().to_excel(writer, index=False, sheet_name='Overview')
    with pd.ExcelWriter(summary_path) as writer:
        summary_stats = data.describe().reset_index()
        summary_stats.rename(columns={'index': 'Statistics'}, inplace=True)
        summary_stats.to_excel(writer, index=False, sheet_name='Summary')

def missing_data_analysis(data, output_dir):
    '''Plot a heatmap to visualize the missing data in the dataset and save the plot.'''

    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), cmap='viridis', cbar=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.title('Missing Data Heatmap')
    plt.savefig(os.path.join(output_dir, 'missing_data_heatmap.png'))
    plt.close()

def correlation_analysis(data, output_dir):
    '''Plot the correlation matrix heatmap for the numeric data in the dataset and save the plot.'''

    numeric_data = data.drop(columns=['Country Name', 'Time'])  # Updated column name
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(16, 16))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, center=0, square=True)
    plt.title('Correlation Matrix Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_matrix_heatmap.png'))
    plt.close()

def time_series_analysis(data, country, indicators, output_dir):
    '''Plot the time series data for a specific country and selected indicators and save the plots.'''

    for indicator in indicators:
        indicator_data = data[data['Country Name'] == country][['Time', indicator]]  # Updated column name

        if indicator_data.empty:
            print(f"No data available for {indicator} in {country}. Skipping...")
            continue
        _, ax = plt.subplots(figsize=(14, 8)) # _ for unused variable
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
        plt.savefig(os.path.join(output_dir, f'time_series_{indicator}.png'))
        plt.close()

def distribution_analysis(data, country, target_indicators, output_dir):
    '''Plot the distribution of target indicators for a specific country and save the plots.'''

    for series_name, label in target_indicators.items():
        if series_name not in data.columns:
            print(f"{series_name} is not present in the dataset. Skipping...")
            continue
        indicator_data = data[data['Country Name'] == country][['Country Name', 'Time', series_name]]  # Updated
        values = indicator_data[series_name].values.flatten()
        values = values[~np.isnan(values)]
        plt.figure(figsize=(10, 6))
        sns.histplot(values, kde=True, bins=30)
        plt.xlabel(label)
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {label} for {country}')
        plt.savefig(os.path.join(output_dir, f'distribution_{series_name}.png'))
        plt.close()

def analyze(df, selected_country, selected_indicator, output_dir):
    """Performs EDA on the provided DataFrame and saves the plots."""

    sns.set(style='whitegrid')
    data_summary_overview(df, output_dir)
    missing_data_analysis(df, output_dir)
    correlation_analysis(df, output_dir)
    time_series_analysis(df, selected_country, [selected_indicator[0]], output_dir)
    distribution_analysis(df, selected_country, {selected_indicator[0]: selected_indicator[0]}, output_dir)
    return "EDA completed."