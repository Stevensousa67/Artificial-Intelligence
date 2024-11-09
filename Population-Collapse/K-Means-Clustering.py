''' Authors: Steven Sousa & Nicholas Pace
    Instructor: Dr. Poonam Kumari
    Course: CS470 - Intro to Artificial Intelligence
    Institution: Bridgewater State University 
    Date: 11/07/2024
    Description: This is the K-Means Clustering file for the Population Collapse project. This file has the following objectives:
        1. Read in the data from the .xlsx file.
        2. Perform K-Means clustering on the data to identify patterns in the features.
    version: 1.0'''

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import utils

def preprocess_data(population_data):
    '''Preprocess the data by handling missing values and scaling the features.'''

    # Handle missing values by imputing with the mean of the numeric columns
    numeric_columns = population_data.select_dtypes(include=['number']).columns
    population_data[numeric_columns] = population_data[numeric_columns].fillna(population_data[numeric_columns].mean())
    return population_data

def perform_kmeans_clustering(population_data, features, k=3):
    '''Perform K-Means clustering on the data using the specified features and number of clusters.'''

    # Select relevant features for clustering
    X = population_data[features]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit the K-Means model
    kmeans = KMeans(n_clusters=k, random_state=42)  # Set random_state for reproducibility
    kmeans.fit(X_scaled)

    # Get the cluster labels for each country
    population_data["Cluster"] = kmeans.labels_

    return population_data, X_scaled, kmeans

def visualize_clusters(X_scaled, kmeans):
    '''Visualize the clusters using a scatter plot of the scaled features.'''

    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis')
    plt.xlabel("Age Dependency Ratio (Scaled)")
    plt.ylabel("Fertility Rate (Scaled)")
    plt.title("K-Means Clustering of Countries")
    plt.show()

def analyze_clusters(population_data, k):
    '''Analyze the clusters by printing summary statistics for each cluster.'''

    for i in range(k):
        cluster_data = population_data[population_data["Cluster"] == i]
        print(f"Cluster {i}:")
        print(cluster_data[["Country", "Age dependency ratio (% of working-age population)", "Fertility rate, total (births per woman)"]].describe())
        print()

def main():
    '''Main function to perform K-Means clustering on the population data.'''
    
    population_data = utils.load_data(utils.FILE_PATH)
    if population_data is None:
        return

    population_data = preprocess_data(population_data)

    features = ["Age dependency ratio (% of working-age population)", "Fertility rate, total (births per woman)"]
    k = 3

    population_data, X_scaled, kmeans = perform_kmeans_clustering(population_data, features, k)
    visualize_clusters(X_scaled, kmeans)
    analyze_clusters(population_data, k)

if __name__ == "__main__":
    main()