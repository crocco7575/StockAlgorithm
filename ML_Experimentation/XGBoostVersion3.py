import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
import time

# Load data
df = pd.read_excel('.venv/ml_shuffled.xlsx')

# Convert 'P/L (%)' to binary label
df['P/L (%)'] = df['P/L (%)'].apply(lambda x: 0 if x < -2.1 else 1)

# Convert non-numeric columns to numeric
numeric_columns = ['Flag Price', 'Buy Price', 'Time Between Flag and Buy', 'Lowest RSI', 'TTM Strength']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with non-numeric values
df = df.dropna()

# Split data into features (X) and target (y)
columns_to_drop = ['P/L (%)', 'P/L ($)', 'Flag Time', 'Sell Time', 'Ticker', 'Buy Time']
X = df.drop(columns_to_drop, axis=1)
y = df['P/L (%)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

# Function to train and evaluate model
def train_and_evaluate(X_train, X_test, y_train, y_test, use_smote=False, use_all_features=True):
    if not use_all_features:
        X_train = X_train[['TTM Strength']]
        X_test = X_test[['TTM Strength']]
    
    # # Scale features
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    
    # Hyperparameter tuning
    param_dist = {
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 500, 1000],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
    }
    
    clf = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist", device="cpu")
    
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=10, cv=3, random_state=42, n_jobs=-1)
    
    start_time = time.time()
    random_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    best_model = random_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Model trained in {training_time:.2f} seconds")
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    if use_all_features:
        importance = best_model.feature_importances_
        for i, col in enumerate(X_train.columns):
            print(f"{col}: {importance[i]}")
    
    return best_model, accuracy, roc_auc

# Train and evaluate models
print("Model with all features:")
model_all, acc_all, roc_all = train_and_evaluate(X_train, X_test, y_train, y_test, use_smote=True, use_all_features=True)

print("\nModel with only TTM Strength:")
model_ttm, acc_ttm, roc_ttm = train_and_evaluate(X_train, X_test, y_train, y_test, use_smote=True, use_all_features=False)

print(f"\nAccuracy difference: {abs(acc_all - acc_ttm):.4f}")
print(f"ROC AUC difference: {abs(roc_all - roc_ttm):.4f}")