import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf

# Load and preprocess data
data = pd.read_csv('data.csv')
data.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})

# Split data
X = data.drop(['diagnosis'], axis=1)
y = data['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2019)

# Visualize data
sns.countplot(data['diagnosis'])
plt.show()

fig = px.scatter_matrix(data, dimensions=X.columns, color='diagnosis')
fig.show()

# Model training and evaluation
models = {
    'SVM': SVC(gamma='scale'),
    'DecisionTree': DecisionTreeClassifier(),
    'LogisticRegression': LogisticRegression(solver='lbfgs', max_iter=2000),
    'RandomForest': RandomForestClassifier(n_estimators=100),
    'TensorFlow': None,  # Placeholder for TensorFlow model
    'PyTorch': None  # Placeholder for PyTorch model
}

print("Model Evaluation:")
for name, model in models.items():
    if name in ['SVM', 'DecisionTree', 'LogisticRegression', 'RandomForest']:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f'{name} Accuracy: {acc:.2f}')
        print(f'{name} Classification Report:\n{report}')
        print(f'{name} Confusion Matrix:\n{cm}')
    elif name == 'TensorFlow':
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(X.columns),)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        tf_model.fit(X_train, y_train, epochs=20, verbose=0)
        tf_acc = tf_model.evaluate(X_test, y_test)[1]
        y_pred_tf = (tf_model.predict(X_test) >= 0.5).astype(int)
        report_tf = classification_report(y_test, y_pred_tf)
        cm_tf = confusion_matrix(y_test, y_pred_tf)
        print(f'TensorFlow Accuracy: {tf_acc:.2f}')
        print(f'TensorFlow Classification Report:\n{report_tf}')
        print(f'TensorFlow Confusion Matrix:\n{cm_tf}')
    elif name == 'PyTorch':
        class PyTorchNN(nn.Module):
            def __init__(self, input_dim):
                super(PyTorchNN, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 1)

            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = torch.sigmoid(self.fc3(x))
                return x

        input_dim = len(X.columns)
        pytorch_model = PyTorchNN(input_dim)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

        X_train_torch = torch.tensor(X_train.values, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_torch = torch.tensor(X_test.values, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        for epoch in range(20):
            optimizer.zero_grad()
            outputs = pytorch_model(X_train_torch)
            loss = criterion(outputs, y_train_torch)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pytorch_outputs = pytorch_model(X_test_torch)
            y_pred_pytorch = (pytorch_outputs >= 0.5).float()
            acc_pytorch = accuracy_score(y_test, y_pred_pytorch)
            report_pytorch = classification_report(y_test, y_pred_pytorch)
            cm_pytorch = confusion_matrix(y_test, y_pred_pytorch)
            print(f'PyTorch Accuracy: {acc_pytorch:.2f}')
            print(f'PyTorch Classification Report:\n{report_pytorch}')
            print(f'PyTorch Confusion Matrix:\n{cm_pytorch}')

# Additional Tests and Analysis
print("\nAdditional Tests and Analysis:")
for name, model in models.items():
    if name in ['SVM', 'DecisionTree', 'LogisticRegression', 'RandomForest']:
        print(f'\n{name} Additional Analysis:')
        for feature in X.columns:
            print(f'Feature: {feature}')
            feature_values = X[feature]
            print(f'Min: {feature_values.min()}, Max: {feature_values.max()}, Mean: {feature_values.mean()}, Std: {feature_values.std()}')
    elif name == 'TensorFlow':
        # Additional analysis for TensorFlow can be added here
        pass
    elif name == 'PyTorch':
        # Additional analysis for PyTorch can be added here
        pass

# Regression Analysis
print("\nRegression Analysis:")
for feature in X.columns:
    plt.figure(figsize=(10, 6))
    sns.regplot(x=X[feature], y=y, logistic=True, ci=None)
    plt.xlabel(feature)
    plt.ylabel('Diagnosis')
    plt.title(f'Logistic Regression: {feature} vs Diagnosis')
    plt.show()

# Final thoughts and conclusion
print("\nFinal Thoughts and Conclusion:")
# Add your final thoughts and conclusion here

# Prediction Testing
print("\nPrediction Testing:")

# Prepare a test sample (use your own data here)
test_sample = X.sample(n=1, random_state=42)
print("Test Sample:")
print(test_sample)

# Use the best model for prediction
best_model = RandomForestClassifier(n_estimators=100)
best_model.fit(X, y)
prediction = best_model.predict(test_sample)
prediction_label = "Malignant" if prediction[0] == 1 else "Benign"
print(f"Best Model Prediction: {prediction_label}")

# Use the TensorFlow model for prediction
tf_prediction = tf_model.predict(test_sample)
tf_prediction_label = "Malignant" if tf_prediction[0][0] >= 0.5 else "Benign"
print(f"TensorFlow Prediction: {tf_prediction_label}")

# Use the PyTorch model for prediction
with torch.no_grad():
    pytorch_outputs_test = pytorch_model(torch.tensor(test_sample.values, dtype=torch.float32))
    pytorch_prediction = (pytorch_outputs_test >= 0.5).float()
pytorch_prediction_label = "Malignant" if pytorch_prediction[0][0] == 1 else "Benign"
print(f"PyTorch Prediction: {pytorch_prediction_label}")
