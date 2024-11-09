import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the Boston Housing dataset and create DataFrame
boston = fetch_openml(name='boston', version=1)
data = pd.DataFrame(data=boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target

print(data.head())

# Add a new column to data frame classifying homes as either "high price" or "low price" based on a defined threshold (median value of homes)
media_value = data['MEDV'].median()
data['HIGH_PRICE'] = (data['MEDV'] > media_value).astype(int)

# Prepare the data for logistic regression
x = data[['RM', 'LSTAT']]
y = data['HIGH_PRICE']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Model Logistic Regression
model = LogisticRegression()
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy:.2f}')

# Create a grid to plot decision boundaries
x_min, x_max = x['RM'].min() - 1, x['RM'].max() + 1
y_min, y_max = x['LSTAT'].min() - 1, x['LSTAT'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Prepare the grid points for prediction as a Dataframe
grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['RM', 'LSTAT'])

# Make predictions on the grid points
predictions = model.predict(grid_points)
predictions = predictions.reshape(xx.shape)

# Plot
plt.contour(xx, yy, predictions, alpha=0.8, cmap='coolwarm')
plt.scatter(x_test['RM'], x_test['LSTAT'], c=y_test, cmap='coolwarm', s=20, edgecolors='k')
plt.title('Boston Housing Dataset')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Lower Status of the Population')
plt.show()