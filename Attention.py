import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow import keras

# Define the input phrase
phrase = "The cat sat on the mat"

# Tokenize the input phrase
tokens = phrase.lower().split()

# Train the word2vec model
word2vec_model = Word2Vec(sentences=[tokens], vector_size=50, window=5, min_count=1, workers=4)

# Get the word2vec embeddings
embeddings = np.array([word2vec_model.wv[word] for word in tokens])

# Define the RNN model
model = keras.Sequential()
model.add(keras.layers.Input(shape=(len(tokens), 50)))
model.add(keras.layers.SimpleRNN(10, return_sequences=True))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(50, activation='relu')))

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(embeddings.reshape(1, len(tokens), 50), embeddings.reshape(1, len(tokens), 50), epochs=100)

# Make predictions
predictions = model.predict(embeddings.reshape(1, len(tokens), 50))

# Plot the embeddings
plt.figure(figsize=(10, 10))

for i, word in enumerate(tokens):
    plt.scatter(embeddings[i, 0], embeddings[i, 1], color='blue')
    plt.text(embeddings[i, 0], embeddings[i, 1], word)

for i, word in enumerate(tokens):
    plt.scatter(predictions[0, i, 0], predictions[0, i, 1], color='red')
    plt.text(predictions[0, i, 0], predictions[0, i, 1], word)

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Word Embeddings')
plt.show()