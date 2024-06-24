import math

import pandas as pd
from matplotlib import pyplot as plt


def spiral_xy(i, spiral_num):
    """
    Create the data for a spiral.

    Arguments:
        i runs from 0 to 96
        spiral_num is 1 or -1
    """
    φ = i / 16 * math.pi
    r = 6.5 * ((104 - i) / 104)
    x = (r * math.cos(φ) * spiral_num) / 13 + 0.5
    y = (r * math.sin(φ) * spiral_num) / 13 + 0.5
    return x, y


def spiral(spiral_num):
    return [spiral_xy(i, spiral_num) for i in range(97)]


a = ["A", spiral(1)]
b = ["B", spiral(-1)]

print(a)
print(b)
a_points = a[1]
b_points = b[1]

# Separate the x and y coordinates for 'A' and 'B'
a_x, a_y = zip(*a_points)
b_x, b_y = zip(*b_points)

# Create a figure and axis
fig, ax = plt.subplots()

# Plot the points for 'A' and 'B'
ax.scatter(a_x, a_y, label='A', color='r', marker='o')
ax.scatter(b_x, b_y, label='B', color='b', marker='x')

# Add labels and legend
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()

# Show the plot
plt.show()

# Combine all points into a single list
all_points = a_points + b_points

# Create a DataFrame with four columns
df = pd.DataFrame(all_points, columns=['x', 'y'])

# Create binary columns for 'a' and 'b'
df['a'] = [1 if p in a_points else 0 for p in all_points]
df['b'] = [1 if p in b_points else 0 for p in all_points]

print(df)

df.to_csv('./data/spiral.csv', index=False)
