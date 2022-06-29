import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

hp = pd.read_csv('https://raw.githubusercontent.com/anyoneai/notebooks/main/datasets/house-prices.csv')

# ================================================================================================
# sns.scatterplot(x='SqFt', y='Price', data=hp).set_title('Price vs Square Feet')
# plt.show()

x = hp['SqFt'].values.reshape(-1, 1)  # .values transforms dataframe/series into an n-dimensional array.
# reshape(-1, 1) basically transposes it.
y = hp['Price'].values

"""
Can use an inline formating with :.2f for float format
.intercept_ calls attribute that returns the intercept value of the regression line.
.coef_ returns an array with scalars that ponderate independent variables. [0] to select an int. 
"""
lr1 = LinearRegression()
lr1.fit(x, y)
print(f'Equation: y= {lr1.intercept_:.2f} + {lr1.coef_[0]:.2f} x')

# ================================================================================================
"""
Manually plotting the regression line by calculating "y" values, for Xmin and Xmax in our data, with the previous
regression line.
"""

x1 = x.min()
x2 = x.max()

y1 = (lr1.coef_ * x1 + lr1.intercept_)[0]
y2 = (lr1.coef_ * x2 + lr1.intercept_)[0]

sns.scatterplot(x='SqFt', y='Price', data=hp).set_title('Price vs Square Feet')
sns.lineplot(x=[x1, x2], y=[y1, y2], color='r')

# ================================================================================================
"""
Regression with more than one feature.
Plotted not with Seaborn but with Matplotlib since it's a 3D graph.
"""

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # Creates empty 3-axis representation

x = hp["SqFt"]
y = hp["Bedrooms"]
z = hp["Price"]

ax.set_xlabel("SqFt")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price")

ax.scatter(x, y, z)
# ================================================================================================
# You have to define which are the dependent variables and the independent one.

lr2 = LinearRegression()
x = hp[['SqFt', 'Bedrooms']].values
y = hp['Price'].values
lr2.fit(x, y)

coefs = lr2.coef_
intercept = lr2.intercept_
num_samples = len(x)

x = hp["SqFt"]
y = hp["Bedrooms"]
z = hp["Price"]

# np.tile() repeats an array "n" given times, in both 1 and 0 axis.
            # Takes as arguments: (array to be repeated, (number of row repetitions, number of column repetitions))
# np.sort() orders the array in a particular way. Default: ascending.
# Arrays must be repeated in order to build a Plane.
xs = np.tile(np.sort(x), (num_samples, 1))
ys = np.tile(np.sort(y), (num_samples, 1)).T
zs = xs*coefs[0] + ys*coefs[1] + intercept
print(f"Equation: y = {intercept:.2f} + {coefs[0]:.2f}x1 + {coefs[1]:.2f}x2")

fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')

ax.set_xlabel('SqFt')
ax.set_ylabel('Bedrooms')
ax.set_zlabel('Price')

ax.scatter(x, y, z)
# Now, plot plane with half opacity.
ax.plot_surface(xs, ys, zs, alpha=0.5, color='r')

# ================================================================================================
"""
Regression with polynomial features
"""

rng = np.random.RandomState(1) # Generates a Mersenne Twister (a general-purpose pseudorandom number generator (PRNG))
num_samples = 100
x = 10 * rng.rand(num_samples) # Non-variable random array (100).
x.sort()

y = np.sin(x) + 0.1 * rng.randn(num_samples) # Right side of sum is meant not to have a perfect sinusoidal line.
plt.scatter(x, y)

poly_model = make_pipeline(PolynomialFeatures(7), LinearRegression())
poly_model.fit(x.reshape(-1, 1), y)
yfit = poly_model.predict(x.reshape(-1, 1))

plt.scatter(x, y)
plt.plot(x, yfit, color='red')
plt.show()