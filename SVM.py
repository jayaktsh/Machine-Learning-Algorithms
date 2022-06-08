import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics

# loading data
cancer = datasets.load_breast_cancer(return_X_y=False)
X = cancer.data
Y = cancer.target
# spliting x and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=100)

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, Y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Accuracy
print("Accuracy: ",metrics.accuracy_score(Y_test, y_pred))

# Precision
print("Precision:",metrics.precision_score(Y_test, y_pred))

# Recall
print("Recall:",metrics.recall_score(Y_test, y_pred))

# plotting scatter
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show()


# creating linspace between 10 to 15
xfit = np.linspace(10, 15)

# plotting scatter
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
 
# plot a line between the different sets of data
for m, b, d in [(0.98, 8, 0.33), (0.5, 15, 0.55), (-0.2, 25, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
    color='#AAAAAA', alpha=0.2)
 
plt.xlim(10, 15);
plt.show()
