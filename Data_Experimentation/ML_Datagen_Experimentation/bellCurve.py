import numpy as np
import matplotlib.pyplot as plt


def bell_curve(x, translation):
    sigma = 1.5
    y = np.exp(-(((x-translation)-5)**2) / (2*sigma**2)) / (sigma * np.sqrt(2*np.pi))
    return y


def scale_mapping(n):
    if 1 <= n <= 10:
        return -5 + (n * 8 / 10)
    else:
        return "Input must be between 1 and 10 (inclusive)"


x = np.linspace(0, 10, 100)
rating = int(input("Enter volatility rating: "))
trans = scale_mapping(rating)
y = bell_curve(x, trans)

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Bell Curve')
plt.show()
