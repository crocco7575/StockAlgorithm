import pandas as pd
import numpy as np
from scipy.optimize import fsolve

# Step 1: Read the Excel file
df = pd.read_excel('records_folder/trailing_sl/average.xlsx')

# Step 2: Extract x and y data
x = df['Buy Spacer'].values
y = df['P/L Avg'].values

# Step 3: Fit a 6th degree polynomial to the data
coefficients = np.polyfit(x, y, 10)
polynomial = np.poly1d(coefficients)

# Step 4: Find the derivative of the polynomial
derivative = np.polyder(polynomial)

# Step 5: Find the roots of the derivative (critical points)
# Provide an initial guess for fsolve using the x values
critical_points = fsolve(derivative, x)

# Step 6: Evaluate the second derivative to determine maxima
second_derivative = np.polyder(derivative)
second_derivative_values = second_derivative(critical_points)

# Identify maxima (where second derivative is negative)
maxima = critical_points[second_derivative_values < 0]
maxima_values = polynomial(maxima)

# Step 7: Print the results
for x_max, y_max in zip(maxima, maxima_values):
    print(f"Maximum at x = {x_max:.4f}, y = {y_max:.4f}")
