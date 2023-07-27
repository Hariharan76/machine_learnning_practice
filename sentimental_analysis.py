# https://www.analyticsvidhya.com/blog/2022/07/sentiment-analysis-using-python/
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the IMDB dataset
max_features = 10000  # Number of words to consider as features
max_len = 200  # Maximum length of sequences (truncate longer sequences, pad shorter ones)

# Load the dataset and split it into training and testing sets
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

# Pad the sequences to the same length
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# Define the LSTM model
model = Sequential()

# Embedding layer to convert integer indices to dense vectors
model.add(Embedding(max_features, 128, input_length=max_len))

# LSTM layer with 128 units
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))

# Output layer for binary classification (positive or negative sentiment)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 5

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')
