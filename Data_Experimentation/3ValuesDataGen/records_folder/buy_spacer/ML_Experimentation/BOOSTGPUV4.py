import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import time

# Load data
df = pd.read_excel('.venv/ml_train_file2.xlsx')

# Create additional features
df['Price_Change'] = (df['Buy Price'] - df['Flag Price']) / df['Flag Price']
df['Time_To_Sell'] = df['Sell Time'] - df['Buy Time']
df['Daily_Progress'] = df['Flag Time'] / 390  # Assuming 6.5 hours trading day

# Convert 'P/L (%)' to binary label
df['Target'] = df['P/L (%)'].apply(lambda x: 1 if x >= 0 else 0)

# Select features
features = ['Flag Price', 'Buy Price', 'Flag Time', 'Buy Time', 'Time Between Flag and Buy', 
            'Lowest RSI', 'TTM Strength', 'Price_Change', 'Time_To_Sell', 'Daily_Progress']

X = df[features]
y = df['Target']

# Function to train and evaluate model
def train_and_evaluate(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model
    model = LGBMClassifier(n_estimators=1000, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = model.feature_importances_
    for i, col in enumerate(X.columns):
        print(f"{col}: {importance[i]:.4f}")
    
    return model, scaler

# Train and evaluate model
print("Training LightGBM model:")
start_time = time.time()
lgbm_model, scaler = train_and_evaluate(X, y)
print(f"LightGBM model trained in {time.time() - start_time:.2f} seconds")

# Function to make predictions
def make_predictions(model, scaler, X_new):
    X_scaled = scaler.transform(X_new)
    return model.predict(X_scaled), model.predict_proba(X_scaled)[:, 1]

# Threshold optimization
def optimize_threshold(y_true, y_pred_proba):
    thresholds = np.arange(0, 1, 0.01)
    scores = [accuracy_score(y_true, y_pred_proba > threshold) for threshold in thresholds]
    best_threshold = thresholds[np.argmax(scores)]
    return best_threshold

# Find the optimal threshold
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred_proba = lgbm_model.predict_proba(scaler.transform(X_test))[:, 1]
optimal_threshold = optimize_threshold(y_test, y_pred_proba)
print(f"\nOptimal threshold: {optimal_threshold:.2f}")

# Evaluate with optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
print("\nResults with optimal threshold:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_optimal):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_optimal))