# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 16:44:44 2018

@author: vishawar
"""

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

dataset = sio.loadmat("ex6data2.mat") 

X = dataset["X"]*10
y= dataset["y"]*10

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)

from sklearn.svm import SVC
classifier = SVC(kernel='poly',degree=4,random_state=0)
classifier.fit(X_train,y_train)

predict = classifier.predict(X_test)

result_score = np.mean(predict==y_test)*100

from sklearn.metrics import confusion_matrix
conf = np.float64(confusion_matrix(y_test,predict))

#precision = conf[0,0]/(np.sum(conf[0,:]))
#recall = conf[0,0]/(np.sum(conf[:,0]))

#F_score = (2*precision*recall)/(precision+recall)


#from matplotlib.pyplot import ListedColormap
pos_idx = np.where(y==1)
neg_idx = np.where(y==0)

plt.scatter(X[pos_idx,0],X[pos_idx,1],marker ="+",color="blue")
plt.scatter(X[neg_idx,0],X[neg_idx,1],marker ="x",color="red")


h = .01  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, m_max]x[y_min, y_max].
Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, cmap=plt.cm.Paired)

plt.title("SVM ")
plt.xlabel("Test1 Score")
plt.ylabel("Test2 Score")
plt.show()