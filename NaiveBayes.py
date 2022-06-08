import matplotlib.pyplot as plt
# load the iris dataset
from sklearn.datasets import load_wine
wine = load_wine()
 
# store the feature matrix (X) and response vector (y)
X = wine.data
y = wine.target
 
# splitting X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
 
# training the model on training set
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
 
# making predictions on the testing set
y_pred = gnb.predict(X_test)
 
# comparing actual response values (y_test) with predicted response values (y_pred)
from sklearn import metrics
print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred)*100)

# representation
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring');
plt.show()
