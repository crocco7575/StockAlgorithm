import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import joblib
import time

def load_and_preprocess_data(file_path):
    df = pd.read_excel(file_path)
    df['P/L (%)'] = df['P/L (%)'].apply(lambda x: 0 if x < -2.1 else 1)
    
    numeric_columns = ['Flag Price', 'Buy Price', 'Time Between Flag and Buy', 'Lowest RSI', 'TTM Strength']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    columns_to_drop = ['P/L (%)', 'P/L ($)', 'Flag Time', 'Buy Time', 'Sell Time', 'Ticker']
    X = df.drop(columns_to_drop, axis=1)
    y = df['P/L (%)']
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_xgboost(X_train, y_train, X_test, y_test, num_boost_round=5000):
    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_dict = dict(zip(np.unique(y_train), class_weights))
    sample_weights = np.array([weight_dict[y] for y in y_train])

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=10,
        learning_rate=0.05,
        tree_method="hist",
        scale_pos_weight=class_weights[1] / class_weights[0],  # Adjust for class imbalance
        use_label_encoder=False,
        eval_metric='auc'
    )
    
    start_time = time.time()
    clf.fit(X_train, y_train, 
            sample_weight=sample_weights, 
            eval_set=[(X_test, y_test)],
            verbose=100)  # Print progress every 100 iterations
    print(f"\nModel trained in {time.time() - start_time:.2f} seconds")
    
    return clf

def evaluate_model(model, X_test, y_test):
    predictions_proba = model.predict_proba(X_test)[:, 1]
    predictions = (predictions_proba > 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    total_samples = tn + fp + fn + tp
    class_0_samples = tn + fp
    class_1_samples = fn + tp
    
    precision_class_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    auc_roc = roc_auc_score(y_test, predictions_proba)
    
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print(f"\nTotal samples: {total_samples}")
    print(f"Class 0 (P/L < -2.1%) samples: {class_0_samples}")
    print(f"Class 1 (P/L >= -2.1%) samples: {class_1_samples}")
    print(f"Precision for class 0 (P/L < -2.1%): {precision_class_0:.4f}")
    print(f"Recall for class 0 (P/L < -2.1%): {recall_class_0:.4f}")
    print(f"Precision for class 1 (P/L >= -2.1%): {precision_class_1:.4f}")
    print(f"Recall for class 1 (P/L >= -2.1%): {recall_class_1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data('.venv/ml_train_file1.xlsx')
    model = train_xgboost(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)
    
    print("Saving the model...")
    joblib.dump(model, 'xgboost_model.joblib')
    print("Model saved!")