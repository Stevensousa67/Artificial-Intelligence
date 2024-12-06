import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    # Load the California Housing dataset and create DataFrame
    california_housing = fetch_california_housing()
    X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    Y = pd.Series(california_housing.target)

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.33, random_state=42)

    # Use KNN for regression to predict housing prices based on features
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    # Predict housing prices using KNN
    y_pred_knn = knn_model.predict(X_test)

    # MSE for KNN model
    print(f'MSE KNN Model: {mean_squared_error(y_test, y_pred_knn)}')

    # Use KMeans for clustering based on Latitude and Longitude
    lat_lon = X[['Latitude', 'Longitude']]

    # Create and fit the KMeans model to segment the dataset based on location
    kmeans_model = KMeans(n_clusters=3, random_state=42)
    kmeans_model.fit(lat_lon)

    # Predict cluster labels for each data point
    clusters = kmeans_model.predict(lat_lon)

    # Show cluster labels
    print(f'Cluster labels: {clusters[:]}')

    # Add cluster labels to the dataframe for visualization
    X['Cluster'] = clusters

    # Plot scatter plot of clusters based on Latitude and Longitude
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=X, x='Longitude', y='Latitude', hue='Cluster', palette='deep', s=50)
    plt.title('KMeans Clusters based on Latitude and Longitude')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()