import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import time

# Load data
df = pd.read_excel('.venv/ml_train_file2.xlsx')

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
# Calculate permutation feature importance
importance = bst.get_score(importance_type='gain')

# Print feature importance
for feature, score in importance.items():
    print(f"{feature}: {score}")


print(f"\nModel trained in {time.time() - start_time:.2f} seconds")

# Make predictions using the trained model
test_dmat = xgb.DMatrix(X_test)
predictions = bst.predict(test_dmat)

# Convert predictions to binary labels
predictions = [1 if p > -2.1 else 0 for p in predictions]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Overall Accuracy: {accuracy:.4f}")

# # Calculate confusion matrix
# cm = confusion_matrix(y_test, predictions)

# # Calculate accuracy for class 0 (True Negative Rate or Specificity)
# tn, fp, fn, tp = cm.ravel()
# accuracy_class_0 = tn / (tn + fp)
# print(f"Accuracy for class 0 (P/L < -2.1%): {accuracy_class_0:.4f}")

print("Classification Report:")
print(classification_report(y_test, predictions, zero_division=0))

# # Save the model
# print("Saving the model...")
# joblib.dump(bst, '.venv/models/polygon_model.joblib')
# print("Model saved!")