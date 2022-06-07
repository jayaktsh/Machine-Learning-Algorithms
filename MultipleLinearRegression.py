import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, metrics

# Load the dataset
wine = datasets.load_wine(return_X_y=False)

# defining feature matrix x and response vector y
x = wine.data
y = wine.target

# spliting x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=1)

# creating linear regression object
reg = linear_model.LinearRegression()

# training the model using training sets
reg.fit(x_train, y_train)

# regression coefficients
print("Coefficients: ", reg.coef_)

# variance score: 1 means perfect prediction
print("Variance Score: ", reg.score(x_test, y_test))

# plot for residual error

# setting plot style
plt.style.use('fivethirtyeight')
# plotting residual error in training data
plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, color="red", s=15, label='Train Data')
# plotting residual errors in test data
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, color="green", s=15, label='Test Data')
# plotting line for zero residual error
plt.hlines(y=0, xmin=-1, xmax=4, linewidth=2)
# plotting legend
plt.legend(loc='upper right')
#plotting title
plt.title("residual errors")
# showing the plot
plt.show()
