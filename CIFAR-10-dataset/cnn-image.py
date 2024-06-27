import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from datetime import datetime

# Create the pictures folder if it doesn't exist
os.makedirs('pictures', exist_ok=True)

# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
X_train, X_test = X_train / 255.0, X_test / 255.0

# Convert class vectors to binary class matrices
y_train_one_hot = to_categorical(y_train, 10)
y_test_one_hot = to_categorical(y_test, 10)

# Define a CNN model for CIFAR-10
def create_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

cnn_model = create_cnn_model(X_train.shape[1:], 10)
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Measure the training time
start_time = time.time()
cnn_history = cnn_model.fit(X_train, y_train_one_hot, batch_size=64, epochs=100, validation_split=0.1, callbacks=[early_stopping, reduce_lr])
training_time = time.time() - start_time

# Evaluate the CNN model
loss_cnn, acc_cnn = cnn_model.evaluate(X_test, y_test_one_hot, verbose=0)
print(f"Test accuracy (CNN): {acc_cnn*100:.2f}%")
print(f"Training time (CNN): {training_time:.2f} seconds")

# Get CNN model predictions
predictions_cnn = cnn_model.predict(X_test).argmax(axis=1)

# Generate confusion matrix
cm_cnn = confusion_matrix(y_test, predictions_cnn)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_cnn, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - CNN')
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
confusion_matrix_cnn = f"pictures/confusion_matrix_cnn_{timestamp}.png"
plt.savefig(confusion_matrix_cnn)

# Plotting the training time and accuracy
plt.figure(figsize=(10, 6))
epochs = range(1, len(cnn_history.history['loss']) + 1)
plt.plot(epochs, cnn_history.history['loss'], 'r', label='Training loss')
plt.plot(epochs, cnn_history.history['val_loss'], 'b', label='Validation loss')
plt.plot(epochs, cnn_history.history['accuracy'], 'g', label='Training accuracy')
plt.plot(epochs, cnn_history.history['val_accuracy'], 'm', label='Validation accuracy')
plt.title('Training and validation loss/accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
training_validation_plot_file = f"pictures/training_validation_loss_accuracy_{timestamp}.png"
plt.savefig(training_validation_plot_file)

# Save training time information
with open('pictures/training_time.txt', 'w') as f:
    f.write(f"Training time (CNN): {training_time:.2f} seconds\n")
    f.write(f"Test accuracy (CNN): {acc_cnn*100:.2f}%\n")


