import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import os

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
        data.describe().to_excel(writer, index=False, sheet_name='Summary')

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
    time_series_analysis(df, selected_country, [selected_indicator], output_dir)
    distribution_analysis(df, selected_country, {selected_indicator: selected_indicator}, output_dir)
    return "EDA completed."