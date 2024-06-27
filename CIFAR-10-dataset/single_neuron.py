import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import keras_tuner as kt

# Create the pictures folder if it doesn't exist
os.makedirs('pictures', exist_ok=True)

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
if len(tf.config.experimental.list_physical_devices('GPU')) == 0:
    print("No GPU detected. Ensure you have installed TensorFlow and the necessary drivers.")

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert class vectors to binary class matrices
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

# Define a hypermodel for a single neuron NN
def build_model(hp):
    model = Sequential()
    model.add(Flatten(input_shape=(32, 32, 3)))  # Flatten input
    model.add(Dense(
        units=1,  # Single neuron
        activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'])  # Hyperparameter for activation function
    ))
    model.add(Dense(10, activation='softmax'))  # Output layer with softmax activation

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])),  # Hyperparameter for learning rate
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize the tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='my_dir',
    project_name='cifar10_single_neuron_nn'
)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Perform hyperparameter search
tuner.search(X_train, y_train_one_hot, epochs=50, validation_split=0.1, callbacks=[early_stopping, reduce_lr])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the optimal hyperparameters
model = tuner.hypermodel.build(best_hps)

# Measure the training time
start_time = time.time()
history = model.fit(X_train, y_train_one_hot, epochs=100, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
training_time = time.time() - start_time

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Training time: {training_time:.2f} seconds")

# Get model predictions
predictions = model.predict(X_test).argmax(axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Single Neuron NN with Hyperparameter Tuning')
plt.savefig('pictures/confusion_matrix_single_neuron_nn_tuned.png')
plt.show()

# Plotting the training time and accuracy
plt.figure(figsize=(10, 6))
epochs = range(1, len(history.history['loss']) + 1)
plt.plot(epochs, history.history['loss'], 'r', label='Training loss')
plt.plot(epochs, history.history['val_loss'], 'b', label='Validation loss')
plt.plot(epochs, history.history['accuracy'], 'g', label='Training accuracy')
plt.plot(epochs, history.history['val_accuracy'], 'm', label='Validation accuracy')
plt.title('Training and validation loss/accuracy - Single Neuron NN with Hyperparameter Tuning')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig('pictures/training_validation_loss_accuracy_single_neuron_nn_tuned.png')
plt.show()

# Save training time information
with open('pictures/training_time.txt', 'w') as f:
    f.write(f"Training time (Single Neuron NN with Hyperparameter Tuning): {training_time:.2f} seconds\n")
    f.write(f"Test accuracy (Single Neuron NN with Hyperparameter Tuning): {accuracy*100:.2f}%\n")
