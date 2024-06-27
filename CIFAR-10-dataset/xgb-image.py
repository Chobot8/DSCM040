# XGBoost on flattened image data
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=10, eval_metric='mlogloss', use_label_encoder=False)
xgb_model.fit(X_train_flat, y_train.ravel())

# Get XGBoost predictions
predictions_xgb = xgb_model.predict(X_test_flat)

# Evaluate the XGBoost model
acc_xgb = accuracy_score(y_test, predictions_xgb)
loss_xgb = 1 - acc_xgb
print(f"Test accuracy (XGBoost): {acc_xgb*100:.2f}%")

# Generate confusion matrix for XGBoost
cm_xgb = confusion_matrix(y_test, predictions_xgb)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - XGBoost')
plt.savefig('pictures/confusion_matrix_xgb.png')
plt.show()
