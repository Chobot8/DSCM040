import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import psutil
import matplotlib.pyplot as plt
from datetime import datetime

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
    return history, training_time, memory_usage, test_loss, test_accuracy

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
history1, time1, mem1, loss1, acc1 = create_and_train_model(single_neuron_model, X_train, y_train, X_test, y_test)
history2, time2, mem2, loss2, acc2 = create_and_train_model(one_layer_nn_model, X_train, y_train, X_test, y_test)
history3, time3, mem3, loss3, acc3 = create_and_train_model(normal_ffnn_model, X_train, y_train, X_test, y_test)

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
resource_usage_plot_file = f"resource_usage_{timestamp}.png"
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
performance_plot_file = f"performance_{timestamp}.png"
plt.savefig(performance_plot_file)
plt.show()

print(f"Performance plot saved as {performance_plot_file}")
