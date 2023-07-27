# https://www.analyticssteps.com/blogs/convolutional-neural-network-cnn-graphical-visualization-code-explanation
import tensorflow as tf
from tensorflow.keras import layers, models

# Define the CNN model
def create_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output of the last convolutional layer
    model.add(layers.Flatten())

    # Dense layers for classification
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model

# Example usage:
# Assuming you have a dataset with input images of shape (height, width, channels)
# and corresponding labels for classification.

# Replace these values with your actual data dimensions and the number of classes
input_shape = (128, 128, 3)  # Example: (height, width, channels)
num_classes = 10  # Example: number of classes for classification

# Create the CNN model
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Now you can use this model to train and evaluate on your dataset.
# You'll need to prepare your dataset in the appropriate format for training and testing.
# Typically, you'll convert the data to numpy arrays and one-hot encode the labels if they're categorical.

# For training:
# model.fit(x_train, y_train, epochs=10, batch_size=32)

# For evaluation:
# test_loss, test_acc = model.evaluate(x_test, y_test)

# For making predictions:
# predictions = model.predict(x_new_data)
