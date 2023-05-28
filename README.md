# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2.Read the dataset.
3.Define X and Y array.
4.Define a function for costFunction,cost and gradient.
5.Define a function to plot the decision boundary and predict the Regression value. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Manjupriya P
RegisterNumber: 212220220024  
*/
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)


## Output:
1.Array value of x :
![Screenshot (14)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/2cab77f5-55c7-4cbe-be61-b9a89fe19c64))
2.Array value of y :
![Screenshot (15)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/b7c67afc-9dfe-460a-8189-2caeaa2199d2)
3.Exam 1 & 2 score graph :
![Screenshot (16)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/9d41d30c-3de6-4a61-90e5-ecc2fc1c51a6)
4.Sigmoid graph :
![Screenshot (17)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/e7ce2d86-e0d2-4470-b67c-1c2284cba4cc)

5.J and grad value with array[0,0,0] :
![Screenshot (18)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/183b5f12-9fac-4183-829b-a85b0b9ee5ba)
6.J and grad value with array[-24,0.2,0.2] :
![Screenshot (19)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/a56706e3-a938-413f-a268-06bf82b030b8)
7.res.function & res.x value :
![Screenshot (20)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/cb6692a2-3cc5-4e0f-aea6-2b4495c13366)
8.Decision Boundary graph :
![Screenshot (21)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/404970a0-cacc-48c3-a5c7-30db5893fa3d)
9.probability value :
![Screenshot (22)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/2c1bac57-410b-484f-a002-2768b4e08c12)

10.Mean prediction value :
![Screenshot (23)](https://github.com/Manjupriya1207/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/113583090/22179a38-f6f0-4fa9-81b7-bc4bea754aad)





## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

