# Machine-Learning-Algorithms
Machine learning is the science of getting computers to act without being explicitly programmed. There are three types of machine learning systems:
1. Supervised learning
2. Unsupervised learning
3. Reinforcement learning

## Supervised Learning: 
Supervised machine learning models make use of labeled data to learn from and make predictions on unlabeled data. For example, smap filteration. Supervised learning can be grouped in two categories: 1. Classification   2. Regression

## Unsupervied Learning:
In this, the machine learns without and supervision. The goal of unsupervised learning is to restructure the input data into new features or a group of objects with similar patterns. It can be further classifieds into two categories of algorithms: 1. Clustering   2. Association

## Reinforcement Learning:
Reinforcement learning is a feedback-based learning method, in which a learning agent gets a reward for each right action and gets a penalty for each wrong action. The agent learns automatically with these feedbacks and improves its performance. The goal is to get more reward points, and hence improve performance. For example, robotic arm.

# The Common Machine Learning Algorithms are:
## 1. Linear Regression: 
Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) variables, hence called as linear regression. It comes under Supervised Learning. Linear regression can be further divided into two types of the algorithm:
#### 1. Simple Linear Regression:
If a single independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Simple Linear Regression.The output plot of SimpleLinearRegression.py

![simpleLinearRegression](https://user-images.githubusercontent.com/104818574/172608442-363b9940-f516-46ac-aed3-638394ed37c8.png)

#### 2. Multiple Linear regression:
If more than one independent variable is used to predict the value of a numerical dependent variable, then such a Linear Regression algorithm is called Multiple Linear Regression. The output plot of MultipleLinearRegression.py

![MultipleLinearRegression](https://user-images.githubusercontent.com/104818574/172609130-6ea25ee2-8462-4a3a-87e7-5aba2b3ed0b3.png)

## 2. Logistic Regression:
Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1. The output plot of LogisticRegression.py

![logisiticRegression](https://user-images.githubusercontent.com/104818574/172609610-e12d4a14-d900-4456-a3bc-4c2510320516.png)

## 3. Decision Tree Classification Algorithm:
Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome. In a Decision tree, there are two nodes, which are the Decision Node and Leaf Node. Decision nodes are used to make any decision and have multiple branches, whereas Leaf nodes are the output of those decisions and do not contain any further branches.

The Ouput of DecisionTreeClassifier.py

	 	 Results Using Gini Index:

Predicted values:
[0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2 1 2 1 2 2 0 1
 0 1 2 2 0 1 2 1]
 
Confusion Matrix: 
  [[14  0  0]
 [ 0 17  1]
 [ 0  1 12]]
 
Accuracy :  95.55555555555556

Report:              precision      recall      f1-score      support

           0       1.00      1.00      1.00        14
           1       0.94      0.94      0.94        18
           2       0.92      0.92      0.92        13

    accuracy                           0.96        45
   macro avg       0.96      0.96      0.96        45
   
weighted avg       0.96      0.96      0.96        45

	 	 Results Using Entropy: 

Predicted values:
[0 1 1 0 2 1 2 0 0 2 1 0 2 1 1 0 1 1 0 0 1 1 2 0 2 1 0 0 1 2 1 2 1 2 2 0 1
 0 1 2 2 0 1 2 1]
 
Confusion Matrix: 
  [[14  0  0]
 [ 0 17  1]
 [ 0  1 12]]
 
Accuracy :  95.55555555555556

Report:                 precision         recall      f1-score       support

           0       1.00      1.00      1.00        14
           1       0.94      0.94      0.94        18
           2       0.92      0.92      0.92        13

    accuracy                           0.96        45
   macro avg       0.96      0.96      0.96        45
   
weighted avg       0.96      0.96      0.96        45

## 4. Naive Bayes Classification Algorithm:
Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems. It is mainly used in text classification that includes a high-dimensional training dataset. Some popular examples of Naïve Bayes Algorithm are spam filtration, Sentimental analysis, and classifying articles. The output plot of NaiveBAyes.py 

![NaiveBayes](https://user-images.githubusercontent.com/104818574/172611640-653b58a2-5fcf-43d2-9aae-b5896702a438.png)


## 5. Support Vector MAchine Algorithm: 
Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane. SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. The output plot of SVM.py

![Svm1](https://user-images.githubusercontent.com/104818574/172612525-7ffe1c9f-e0fb-451b-b678-f57b7533bc28.png)

![svm2](https://user-images.githubusercontent.com/104818574/172612538-66f1a182-ec44-4876-b80f-63fe7a3f8ce0.png)


## 6. K-NN Algorithm:
K-NN algorithm assumes the similarity between the new case/data and available cases and put the new case into the category that is most similar to the available categories. K-NN algorithm stores all the available data and classifies a new data point based on the similarity. This means when new data appears then it can be easily classified into a well suite category by using K- NN algorithm. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset. The output plot of knn.py

![knn](https://user-images.githubusercontent.com/104818574/172612749-7ab6a4ea-596b-44f9-8298-15e678ec173b.png)
