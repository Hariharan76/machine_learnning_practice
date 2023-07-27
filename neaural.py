import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset and preprocess it
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the pixel values to be in the range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode the target labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Build the neural network
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))  # Flatten the 28x28 input images into a 1D array
model.add(Dense(128, activation='relu'))    # Hidden layer with 128 neurons and ReLU activation
model.add(Dense(10, activation='softmax'))  # Output layer with 10 neurons for each class and softmax activation

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 128
epochs = 10
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

# This code creates a neural network with one hidden layer containing 128 neurons and an output layer with 10 neurons (one for each digit). We use the ReLU activation function in the hidden layer and the softmax activation function in the output layer to produce probabilities for each digit class.

# The model is trained using the Adam optimizer and categorical cross-entropy as the loss function. The data is normalized, and the target labels are one-hot encoded for training. After training, the model is evaluated on the test set to measure its performance.

# This is a basic example to get you started with neural networks in Python. Depending on the problem and dataset, you may need to adjust the network architecture, hyperparameters, and other settings to achieve better results. Additionally, TensorFlow provides various other layers, optimizers, and techniques to improve the model's performance and robustness.
#As of my knowledge cutoff in September 2021, scikit-learn does not have built-in support for deep learning or neural networks. Scikit-learn is primarily focused on traditional machine learning algorithms for tasks like classification, regression, clustering, and dimensionality reduction.

#For neural networks and deep learning in Python, you typically use libraries like TensorFlow, Keras, or PyTorch, which are specifically designed for these purposes. However, if you are looking for a simpler interface to neural networks, you might want to consider using Keras, which is integrated with TensorFlow and provides an easy-to-use API.