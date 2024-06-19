import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

## LSTM imports:
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.utils import to_categorical

# Step 1: Load and Prepare the Wine Quality Dataset
# -----------------------------------------------
# Load the dataset
red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
white_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'
red_wine = pd.read_csv(red_wine_url, sep=';')
white_wine = pd.read_csv(white_wine_url, sep=';')

# Combine the datasets
wine = pd.concat([red_wine, white_wine], axis=0)

# Split into features and target
X = wine.drop('quality', axis=1)
y = wine['quality']

# Adjust labels to be 0-based
y = y - 3

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to create and train a model
def create_and_train_model(model_fn, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    model = model_fn()
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    training_time = end_time - start_time
    memory_usage = end_memory - start_memory
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return model, history, training_time, memory_usage, test_loss, test_accuracy

# Step 2: Define and Train Three Neural Networks
# -----------------------------------------------

# Model 1: Single Neuron Model
def single_neuron_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, input_shape=(X_train.shape[1],), activation='softmax')
    ])
    return model

# Model 2: One Layer Neural Network
def one_layer_nn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, input_shape=(X_train.shape[1],), activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Model 3: Normal Feedforward Neural Network
def normal_ffnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train the models
model_single_neuron, history1, time1, mem1, loss1, acc1 = create_and_train_model(single_neuron_model, X_train, y_train, X_test, y_test)
model_one_layer_neuron,history2, time2, mem2, loss2, acc2 = create_and_train_model(one_layer_nn_model, X_train, y_train, X_test, y_test)
model_normal_neuron, history3, time3, mem3, loss3, acc3 = create_and_train_model(normal_ffnn_model, X_train, y_train, X_test, y_test)

# Step 3: Compare Performance on Resource Usage
# -----------------------------------------------
resource_usage = {
    'Model': ['Single Neuron', 'One Layer NN', 'Normal FFNN'],
    'Training Time (s)': [time1, time2, time3],
    'Memory Usage (bytes)': [mem1, mem2, mem3]
}

resource_usage_df = pd.DataFrame(resource_usage)

# Step 4: Visualize Resource Usage
# -----------------------------------------------
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.bar(resource_usage_df['Model'], resource_usage_df['Training Time (s)'], color='g')
ax2.plot(resource_usage_df['Model'], resource_usage_df['Memory Usage (bytes)'], color='b')

ax1.set_xlabel('Model')
ax1.set_ylabel('Training Time (s)', color='g')
ax2.set_ylabel('Memory Usage (bytes)', color='b')

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
resource_usage_plot_file = f"pictures/resource_usage_{timestamp}.png"
plt.savefig(resource_usage_plot_file)
# plt.show()

# Step 5: Compare Performance on End Result
# -----------------------------------------------
performance = {
    'Model': ['Single Neuron', 'One Layer NN', 'Normal FFNN'],
    'Test Loss': [loss1, loss2, loss3],
    'Test Accuracy': [acc1, acc2, acc3]
}

performance_df = pd.DataFrame(performance)

# Step 6: Visualize Results
# -----------------------------------------------
# Ensure correct data types
performance_df['Test Loss'] = performance_df['Test Loss'].astype(float)
performance_df['Test Accuracy'] = performance_df['Test Accuracy'].astype(float)

# Plotting Test Loss and Test Accuracy separately
fig, ax1 = plt.subplots()

# Plot Test Loss
color = 'tab:red'
ax1.set_xlabel('Model')
ax1.set_ylabel('Test Loss', color=color)
ax1.bar(performance_df['Model'], performance_df['Test Loss'], color=color)
ax1.tick_params(axis='y', labelcolor=color)

# Create a secondary y-axis to plot Test Accuracy
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Test Accuracy', color=color)
ax2.plot(performance_df['Model'], performance_df['Test Accuracy'], color=color, marker='o', linestyle='-')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and show plot
plt.title('Model Performance Comparison')

# Save the plot with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
performance_plot_file = f"pictures/performance_{timestamp}.png"
plt.savefig(performance_plot_file)
# plt.show()

print(f"Performance plot saved as {performance_plot_file}")


## Create confusion matrix for the Models


# Assuming you have trained models and test data
# Replace these with your actual model and test data variables
model1 = model_single_neuron  # Your trained Single Neuron model
model2 = model_one_layer_neuron  # Your trained One Layer NN Model
model3 = model_normal_neuron  # Your trained Normal FFNN model
# X_test =   # Your test feature data
# y_test = ...  # Your test true labels

# Get predictions from your models
predictions = {
    'Single Neuron': model1.predict(X_test).argmax(axis=1),
    'One Layer NN Model': model2.predict(X_test).argmax(axis=1),
    'Normal FFNN': model3.predict(X_test).argmax(axis=1)
}

# Get test loss and accuracy for each model
loss1, acc1 = model1.evaluate(X_test, y_test, verbose=0)
loss2, acc2 = model2.evaluate(X_test, y_test, verbose=0)
loss3, acc3 = model3.evaluate(X_test, y_test, verbose=0)

models = ['Single Neuron', 'One Layer NN Model', 'Normal FFNN']
test_loss = [loss1, loss2, loss3]
test_accuracy = [acc1, acc2, acc3]

# Determine your class labels
class_labels = sorted(np.unique(y_test))  # Automatically get the unique classes from y_test

fig, ax = plt.subplots(2, 2, figsize=(12, 10))

# Flattening the array of axes for easier indexing
ax = ax.flatten()

# Performance plot
ax1 = ax[0]
ax1.set_xlabel('Model')
ax1.set_ylabel('Test Loss', color='tab:red')
ax1.bar(models, test_loss, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('Test Accuracy', color='tab:blue')
ax1_twin.plot(models, test_accuracy, color='tab:blue', marker='o')
ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('Model Performance Comparison')

# Confusion matrices
for i, model in enumerate(models):
    cm = confusion_matrix(y_test, predictions[model])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax[i + 1], colorbar=False)
    ax[i + 1].set_title(f'{model} Confusion Matrix')

plt.tight_layout()
plt.savefig('pictures/enhanced_performance_with_confusion_matrices.png')


# plt.show()
history_df = pd.DataFrame(history1.history)
plt.clf()
# Create the plot
history_df['loss'].plot()
plt.title('Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Save the plot as a figure
plt.savefig('pictures/training_loss_plot.png')

# Show the plot (optional, for interactive environments)
# plt.show()


###################### Add LSTM NN ################################

# Check the maximum value in your input data to determine the vocabulary size
vocab_size = int(np.max([np.max(X_train), np.max(X_test)])) + 1
print(f'Vocabulary size: {vocab_size}')

# Convert labels to categorical (one-hot encoding) if needed
num_classes = len(np.unique(y_train))
y_train_categorical = to_categorical(y_train, num_classes)
y_test_categorical = to_categorical(y_test, num_classes)

# Define and train the LSTM model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=vocab_size, output_dim=128))
lstm_model.add(LSTM(128))
lstm_model.add(Dense(num_classes, activation='softmax'))

lstm_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm_history = lstm_model.fit(X_train, y_train_categorical, epochs=10, batch_size=32, validation_split=0.2)

# Get LSTM model predictions
lstm_predictions = lstm_model.predict(X_test).argmax(axis=1)

# Evaluate the LSTM model
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test_categorical, verbose=0)

# Adding LSTM model to the existing models for comparison
models = ['Single Neuron', 'One Layer NN Model', 'Normal FFNN', 'LSTM NN']
test_loss = [loss1, loss2, loss3, lstm_loss]
test_accuracy = [acc1, acc2, acc3, lstm_acc]

# Predictions dictionary now includes LSTM model predictions
predictions['LSTM NN'] = lstm_predictions

# Define your class labels
class_labels = np.unique(y_test)  # Assuming y_test contains all possible classes

fig, ax = plt.subplots(2, 3, figsize=(18, 10))  # Changed to 2x3 grid

# Flattening the array of axes for easier indexing
ax = ax.flatten()

# Performance plot
ax1 = ax[0]
ax1.set_xlabel('Model')
ax1.set_ylabel('Test Loss', color='tab:red')
ax1.bar(models, test_loss, color='tab:red')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1_twin = ax1.twinx()
ax1_twin.set_ylabel('Test Accuracy', color='tab:blue')
ax1_twin.plot(models, test_accuracy, color='tab:blue', marker='o')
ax1_twin.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('Model Performance Comparison')

# Confusion matrices
for i, model in enumerate(models):
    cm = confusion_matrix(y_test, predictions[model])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(ax=ax[i + 1], colorbar=False)
    ax[i + 1].set_title(f'{model} Confusion Matrix')

plt.tight_layout()
plt.savefig('pictures/enhanced_performance_with_confusion_matrices.png')
# plt.show()

# Save the training loss plot
history_df = pd.DataFrame(lstm_history.history)
plt.figure()
history_df['loss'].plot()
plt.title('LSTM Training Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('pictures/lstm_training_loss_plot.png')
# plt.show()