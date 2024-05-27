import matplotlib.pyplot as plt

# Create a figure and axis
fig, ax = plt.subplots()

# Plot some data (just a straight line in this case)
ax.plot([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])

# Draw a vertical line at x=2
ax.plot([2, 2], [0, 5], color='r', linestyle='--', linewidth=2)

# Show the plot
plt.show()