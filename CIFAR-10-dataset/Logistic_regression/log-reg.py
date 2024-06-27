import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import time
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.datasets import cifar10

# Create the pictures folder if it doesn't exist
os.makedirs('pictures', exist_ok=True)

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Flatten the input images
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Normalize pixel values to be between 0 and 1
scaler = StandardScaler()
X_train_flat = scaler.fit_transform(X_train_flat)
X_test_flat = scaler.transform(X_test_flat)

# Convert class vectors to 1D
y_train = y_train.flatten()
y_test = y_test.flatten()

# Define a logistic regression model
log_reg = LogisticRegression(max_iter=1000, solver='saga', multi_class='multinomial')

# Define hyperparameters to tune
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],
    'penalty': ['l1', 'l2', 'elasticnet'],
    'l1_ratio': [0, 0.5, 1]  # Only used for 'elasticnet' penalty
}

# Initialize GridSearchCV
grid_search = GridSearchCV(log_reg, param_grid, cv=3, n_jobs=-1, verbose=1)

# Measure the training time
start_time = time.time()
grid_search.fit(X_train_flat, y_train)
training_time = time.time() - start_time

# Get the best model
best_log_reg = grid_search.best_estimator_

# Evaluate the model
y_pred = best_log_reg.predict(X_test_flat)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {accuracy*100:.2f}%")
print(f"Training time: {training_time:.2f} seconds")

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Logistic Regression with Hyperparameter Tuning')
plt.savefig('pictures/confusion_matrix_logistic_regression_tuned.png')
# plt.show()

# Plotting the training time and accuracy
# Not applicable for logistic regression as we don't have epoch-based training history

# Save training time information
with open('pictures/training_time_log-reg.txt', 'w') as f:
    f.write(f"Training time (Logistic Regression with Hyperparameter Tuning): {training_time:.2f} seconds\n")
    f.write(f"Test accuracy (Logistic Regression with Hyperparameter Tuning): {accuracy*100:.2f}%\n")
