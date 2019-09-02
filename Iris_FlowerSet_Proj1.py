#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataset
from sklearn.datasets import load_iris
dataset = load_iris()

#initializing variables
X = dataset.data
y = dataset.target

#scatter plot when sepal length and sepal width are taken
plt.scatter(X[:,0],X[:,1])
plt.show()

#scatter plot when petal length and petal width are taken
plt.scatter(X[:,2],X[:,3])
plt.show()

#scatter plot distinguishing all 3 based on Sepal parameters
plt.scatter(X[y==0, 0], X[y==0, 1], c = 'r', label = "Setosa")
plt.scatter(X[y==1, 0], X[y==1, 1], c = 'g', label = "Versicolor")
plt.scatter(X[y==2, 0], X[y==2, 1], c = 'b', label = "Verginica")
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()   #dont know its use
plt.title("Analysis of Iris dataset")
plt.show()

#scatter plot distinguishing all 3 based on Petal parameters
plt.scatter(X[y==0, 2], X[y==0, 3], c = 'r', label = "Setosa")
plt.scatter(X[y==1, 2], X[y==1, 3], c = 'g', label = "Versicolor")
plt.scatter(X[y==2, 2], X[y==2, 3], c = 'b', label = "Verginica")
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()   #dont know its use
plt.title("Analysis of Iris dataset")
plt.show()