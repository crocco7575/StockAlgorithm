import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

# Load your data into a pandas DataFrame
df = pd.read_excel('combined_file.xlsx')

# Convert the 'P/L (%)' column to a binary label (0 for "bad buy", 1 for "good buy")
df['P/L (%)'] = df['P/L (%)'].apply(lambda x: 1 if x > 0 else 0)

# Convert non-numeric columns to numeric
df['Flag Price'] = pd.to_numeric(df['Flag Price'], errors='coerce')
df['Buy Price'] = pd.to_numeric(df['Buy Price'], errors='coerce')
df['Time Between Flag and Buy'] = pd.to_numeric(df['Time Between Flag and Buy'], errors='coerce')
df['Lowest RSI'] = pd.to_numeric(df['Lowest RSI'], errors='coerce')
df['TTM Strength'] = pd.to_numeric(df['TTM Strength'], errors='coerce')

# Remove any rows with non-numeric values
df = df.dropna()

# Split your data into features (X) and target (y)
X = df.drop(['P/L (%)', 'P/L ($)', 'Flag Time', 'Buy Time', 'Sell Time', 'Ticker'], axis=1)
y = df['P/L (%)']

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest classifier
clf = RandomForestClassifier(n_estimators=4600, random_state=42)

# Train the classifier on the training data
clf.fit(X_train, y_train)
print("done with training \n")

# Save the model
#dump(clf, 'random_forest_model.joblib')

# Load the test data
test_data = pd.read_excel('ml_testdata.xlsx')

# Convert non-numeric columns to numeric
test_data['Flag Price'] = pd.to_numeric(test_data['Flag Price'], errors='coerce')
test_data['Buy Price'] = pd.to_numeric(test_data['Buy Price'], errors='coerce')
test_data['Time Between Flag and Buy'] = pd.to_numeric(test_data['Time Between Flag and Buy'], errors='coerce')
test_data['Lowest RSI'] = pd.to_numeric(test_data['Lowest RSI'], errors='coerce')
test_data['TTM Strength'] = pd.to_numeric(test_data['TTM Strength'], errors='coerce')

# Remove any rows with non-numeric values
test_data = test_data.dropna()

# Convert 'P/L (%)' to binary values (0s and 1s)
test_data['P/L (%)'] = test_data['P/L (%)'].apply(lambda x: 1 if x > 0 else 0)

# Make predictions on the test data
predictions = clf.predict(test_data.drop(['P/L (%)', 'P/L ($)', 'Flag Time', 'Buy Time', 'Sell Time', 'Ticker'], axis=1))

# Print the predictions
print("Predictions:", predictions)

# Evaluate the classifier's performance
accuracy = accuracy_score(test_data['P/L (%)'], predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(test_data['P/L (%)'], predictions))