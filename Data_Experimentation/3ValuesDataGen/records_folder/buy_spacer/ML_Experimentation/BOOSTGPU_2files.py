import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time

# Load data
df_train = pd.read_excel('.venv/ml_train_file2.xlsx')
df_test = pd.read_excel('.venv/ml_shuffled.xlsx')

# Convert 'P/L (%)' to binary label for both datasets
df_train['P/L (%)'] = df_train['P/L (%)'].apply(lambda x: 0 if x < -2.1 else 1)
df_test['P/L (%)'] = df_test['P/L (%)'].apply(lambda x: 0 if x < -2.1 else 1)

# Convert non-numeric columns to numeric for both datasets
numeric_columns = ['Flag Price', 'Buy Price', 'Time Between Flag and Buy', 'Lowest RSI', 'TTM Strength']
for col in numeric_columns:
    df_train[col] = pd.to_numeric(df_train[col], errors='coerce')
    df_test[col] = pd.to_numeric(df_test[col], errors='coerce')

# Remove rows with non-numeric values for both datasets
df_train = df_train.dropna()
df_test = df_test.dropna()

# Split data into features (X) and target (y) for both datasets
columns_to_drop = ['P/L (%)', 'P/L ($)', 'Flag Time', 'Buy Time', 'Sell Time', 'Ticker']
X_train = df_train.drop(columns_to_drop, axis=1)
y_train = df_train['P/L (%)']
X_test = df_test.drop(columns_to_drop, axis=1)
y_test = df_test['P/L (%)']

# Initialize XGBoost classifier
clf = xgb.XGBClassifier(objective="binary:logistic", max_depth=10, learning_rate=0.05, tree_method="hist", device="cuda")

# Define the training parameters
params = clf.get_xgb_params()
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
num_boost_round = 5000

# Define the progress callback function
class ProgressCallback(xgb.callback.TrainingCallback):
    def __init__(self):
        self.iteration = 0

    def after_iteration(self, model, epoch, evals_log):
        self.iteration += 1
        print(f"\rIteration {self.iteration}/{num_boost_round} | Metrics: {evals_log}", end="")

# Train the model with a progress bar
start_time = time.time()
callback = ProgressCallback()
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    callbacks=[callback]
)
print(f"\nModel trained in {time.time() - start_time:.2f} seconds")

# Make predictions using the trained model
test_dmat = xgb.DMatrix(X_test)
predictions = bst.predict(test_dmat)

# Convert predictions to binary labels
predictions = [1 if p > 0.5 else 0 for p in predictions]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Overall Accuracy: {accuracy:.4f}")

# Calculate confusion matrix
cm = confusion_matrix(y_test, predictions)

# Calculate accuracy for class 0 (True Negative Rate or Specificity)
tn, fp, fn, tp = cm.ravel()
accuracy_class_0 = tn / (tn + fp)
print(f"Accuracy for class 0 (P/L < -2.1%): {accuracy_class_0:.4f}")

print("Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# # Save the model
# print("Saving the model...")
# joblib.dump(bst, 'xgboost_model.joblib')
# print("Model saved!")