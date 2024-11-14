# This code will perform a RNN using tensorflow keras on the iris dataset

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from tensorflow import keras
import numpy

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# One-hot encode the target labels
encoder = OneHotEncoder(sparse_output=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))
y_test = encoder.transform(y_test.reshape(-1, 1))

# Create the RNN model
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(10, input_shape=(4, 1)))
model.add(keras.layers.Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train.reshape(-1, 4, 1), y_train, epochs=100)

# Make predictions
predictions = model.predict(X_test.reshape(-1, 4, 1))
predictions = numpy.argmax(predictions, axis=1)

# Calculate the accuracy
accuracy = accuracy_score(numpy.argmax(y_test, axis=1), predictions)
print('Accuracy:', accuracy)