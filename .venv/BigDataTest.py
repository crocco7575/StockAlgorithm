import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the CSV file
data = pd.read_csv('2024-05-15.csv')

# Create a matplotlib figure and axis
fig, ax = plt.subplots()

# Plot the data as a scatter plot
ax.scatter(data['TTM Strength'], data['P/L (%)'], c=['green' if p > 0 else 'red' for p in data['P/L (%)']])

# Set labels and title
ax.set_xlabel('TTM Strength')
ax.set_ylabel('Profit/Loss Percent')
ax.set_title('Profit/Loss Percent vs TTM Strength')
# Set the x-axis to start at 0
ax.set_xlim(left=0)
# Set the x-axis to zero
ax.spines['bottom'].set_position('zero')

# Create the pngbigdata folder if it doesn't exist
if not os.path.exists('pngbigdata'):
    os.makedirs('pngbigdata')

# Save the graph as a PNG file in the pngbigdata folder
plt.savefig('pngbigdata/TTM Strength_vs_profit_loss.png')