#!/usr/bin/env python
# coding: utf-8
# In[1]:
"""Original file is located at
  

 https://colab.research.google.com/drive/1GCc7CgwWuJgZXIgFnPxwX_12UMysJ7vO
 """


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load the Iris dataset
iris = load_iris()
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# Create a decision tree classifier
clf = DecisionTreeClassifier()
# Train the classifier on the training data
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#Evaluate the accuracy of the model-
accuracy = accuracy_score(y_test, y_pred)
sepallength=float(input("ENTER SEPAL LENGTH OF FLOWER:"))
sepalwidth=float(input("ENTER SEPAL WIDTH OF FLOWER:"))
petallength=float(input("ENTER PETAL LENGTH OF FLOWER:"))
petalwidth=float(input("ENTER PETAL WIDTH OF FLOWER:"))
print("Accuracy:", accuracy)
new_sample = [[sepallength,sepalwidth,petallength,petalwidth]]
predicted_class = clf.predict(new_sample)
predicted_species = iris.target_names[predicted_class]
print("Predicted species:", predicted_species)


# In[ ]:




