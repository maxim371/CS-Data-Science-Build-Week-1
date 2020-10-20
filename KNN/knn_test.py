from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import  numpy as np

iris = datasets.load_iris()
x = iris.data
y = iris.target

k = 3
model = KNeighborsClassifier(n_neighbors = k)
model.fit(x, y)

#Predictions
print('(-3, -3, 3, 3) is class:')
print(model.predict(x, [-3, -3, 3, 3]))

print('(2, 4, 5, 5, 4, 2) is class:')
print(model.predict(x, [2, 4, 5, 5, 4, 2]))