import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# Load data
df = pd.read_excel('.venv/ml_shuffled.xlsx')

# Convert 'P/L (%)' to a single binary label column
df['P/L (%)'] = df['P/L (%)'].apply(lambda x: 
                                    0 if x < -2 else 
                                    1 if x < 0 else 
                                    2 if x < 2 else 
                                    3)

# Convert non-numeric columns to numeric
df['Flag Price'] = pd.to_numeric(df['Flag Price'], errors='coerce')
df['Buy Price'] = pd.to_numeric(df['Buy Price'], errors='coerce')
df['Time Between Flag and Buy'] = pd.to_numeric(df['Time Between Flag and Buy'], errors='coerce')
df['Lowest RSI'] = pd.to_numeric(df['Lowest RSI'], errors='coerce')
df['TTM Strength'] = pd.to_numeric(df['TTM Strength'], errors='coerce')

# Remove rows with non-numeric values
df = df.dropna()

# Split data into features (X) and target (y)
X = df.drop(['P/L (%)', 'P/L ($)', 'Flag Time', 'Buy Time', 'Sell Time', 'Ticker'], axis=1)
y = df['P/L (%)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost classifier
clf = xgb.XGBClassifier(objective="multi:softmax", num_class=4, max_depth=10, learning_rate=0.05, tree_method="hist", device="cuda")

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



# Make predictions using the trained model
predictions = bst.predict(dtest)
predictions = bst.predict(test_dmat)
predictions = [round(value) for value in predictions]

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, predictions))