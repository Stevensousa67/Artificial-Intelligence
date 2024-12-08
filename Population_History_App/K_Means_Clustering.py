''' Authors: Steven Sousa & Nicolas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/08/2024
    version: 1.0
    Description: This is the K-Means Clustering file for the Population History project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Perform K-Means clustering on the data to identify patterns in the features.'''

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os, matplotlib
import pandas as pd

matplotlib.use('Agg')


def preprocess_data(population_data):
    '''Preprocess the data by handling missing values and scaling the features.'''

    # Handle missing values by imputing with the mean of the numeric columns
    numeric_columns = population_data.select_dtypes(include=['number']).columns
    population_data[numeric_columns] = population_data[numeric_columns].fillna(population_data[numeric_columns].mean())
    return population_data


def perform_kmeans_clustering(population_data, features, k=3):
    '''Perform K-Means clustering on the data using the specified features and number of clusters.'''

    # Ensure features is a list
    if isinstance(features, str):
        features = [features]

    # Select relevant features for clustering
    X = population_data[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Check if X_scaled has at least two dimensions
    if X_scaled.shape[1] < 2:
        raise ValueError(f"The selected indicators must result in at least two dimensions. "
                         f"Currently selected features: {features}")

    # Fit the K-Means model
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducibility
    kmeans.fit(X_scaled)

    # Get the cluster labels for each country
    population_data["Cluster"] = kmeans.labels_

    return population_data, X_scaled, kmeans

def visualize_clusters(X_scaled, kmeans, output_dir, selected_country, selected_indicators):
    '''Visualize the clusters using a scatter plot of the scaled features.'''

    plt.figure(figsize=(8, 6))

    # Check if X_scaled has at least two dimensions
    if X_scaled.shape[1] < 2:
        raise ValueError("The selected indicators must result in at least two dimensions for visualization.")

    print(f"Shape of X_scaled in visualize_clusters: {X_scaled.shape}")  # Debug statement

    # Create a mapping of cluster numbers to indicator names
    cluster_labels = [f'{selected_indicators[0]} vs {selected_indicators[1]}' for _ in range(kmeans.n_clusters)]

    # Plot each cluster with a different color and label
    for cluster in range(kmeans.n_clusters):
        plt.scatter(X_scaled[kmeans.labels_ == cluster, 0], X_scaled[kmeans.labels_ == cluster, 1], 
                    label=cluster_labels[cluster], cmap='viridis')

    plt.xlabel(f"{selected_indicators[0]} (Scaled)")
    plt.ylabel(f"{selected_indicators[1]} (Scaled)")
    plt.title(f"K-Means Clustering of {selected_country}")
    plt.legend()  # Add legend to the plot

    # Save the plot to a file
    plt.savefig(os.path.join(output_dir, f'k_means_clustering.png'))
    plt.close()


def analyze_clusters(population_data, k, output_dir):
    '''Analyze the clusters by exporting summary statistics for each cluster to a single Excel sheet.'''
    analysis_path = os.path.join(output_dir, 'cluster_analysis.xlsx')

    all_summary_stats = []

    for i in range(k):
        cluster_data = population_data[population_data["Cluster"] == i]
        print(f"Cluster {i}:")
        summary_stats = cluster_data.describe()
        summary_stats['Cluster'] = i  # Add a column to indicate the cluster number
        all_summary_stats.append(summary_stats)
        print(summary_stats)
        print()

    # Concatenate all summary statistics into a single DataFrame
    combined_summary_stats = pd.concat(all_summary_stats)

    # Write the combined summary statistics to a single sheet in the Excel file
    with pd.ExcelWriter(analysis_path) as writer:
        combined_summary_stats.to_excel(writer, sheet_name='Cluster_Analysis')

    print(f"Cluster analysis exported to {analysis_path}")


def analyze(df, selected_country, selected_indicator, output_dir):
    '''Main function to perform K-Means clustering on the population data.'''

    # Ensure there are at least two indicators
    if len(selected_indicator) < 2:
        all_numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        remaining_features = [col for col in all_numeric_columns if col not in selected_indicator]
        if remaining_features:
            selected_indicator.append(remaining_features[0])
        else:
            raise ValueError("Not enough numeric features available for clustering. " f"Available features: {all_numeric_columns}")

    df = preprocess_data(df)
    k = len(selected_indicator)  # Use the number of selected indicators as the number of clusters
    df, X_scaled, kmeans = perform_kmeans_clustering(df,selected_indicator, k)

    visualize_clusters(X_scaled, kmeans, output_dir, selected_country, selected_indicator)
    analyze_clusters(df, k, output_dir)