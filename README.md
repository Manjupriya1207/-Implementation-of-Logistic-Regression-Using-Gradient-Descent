# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import dataset
2. Assign x and y values
3. Calculate Logistic sigmoid function and plot the graph
4. Calculate the cos function
5. Calculate x train and y train grad value
6. Calculate and Plot decision boundry
7. Calculate the probability value and predict the mean value
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Manjupriya P
RegisterNumber: 212220220024

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
x=data[:,[0,1]]
y=data[:,2]
x[:5]
y[:5]
plt.figure()
plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not admitted")
plt.xlabel("exam 1 score")
plt.ylabel("exam 2 score")
plt.legend()
plt.show()
def sigmoid(z):
  return 1/(1+np.exp(-z))
plt.plot()
x_plot=np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()
def cf(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
j,grad=cf(theta,x_train,y)
print(j)
print(grad)
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=cf(theta,x_train,y)
print(j)
print(grad)
def cost(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j
def gradient(theta,x,y):
  h=sigmoid(np.dot(x,theta))
  grad=np.dot(x.T,h-y)/x.shape[0]
  return grad
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)
def decbou(theta,x,y):
  x_min,x_max=x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max=x[:,0].min()-1,x[:,0].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.01))
  x_plot=np.c_[xx.ravel(),yy.ravel()]
  x_plot=np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot=np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y==1][:,0],x[y==1][:,1],label="Admitted")
  plt.scatter(x[y==0][:,0],x[y==0][:,1],label="Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  decbou(res.x,x,y)
  prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)
def predict(theta,x):
  x_train=np.hstack((np.ones((x.shape[0],1)),x))
  prob=sigmoid(np.dot(x_train,theta))
  return (prob>=0.5).astype(int)
 np.mean(predict(res.x,x)==y)
*/
```

## Output:
## ARRAY VALUE OF X:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/2c8d879c-3eef-43d9-b83c-2c611ebf032f)
## ARRAY VALUE OF Y:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/3ab98a47-8112-47e1-b103-067a0337ddab)
## VISUALIZING OF DATA:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/0bc3569f-7ed2-4b3f-8632-e59c9884c006)
## SIGMOID FUNCTION:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/6a5f2475-fb6e-4abd-bc34-479c741121e0)
## COST FUNCTION:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/ca7c8443-96c1-4b4b-933b-484512c87c3b)
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/1b708f67-4055-40f3-ac59-0e2e97bfade8)
## DECISION BOUNDARY-GRAPH:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/16357b0b-5fd9-4a19-b954-bb2581988e53)
## PROBABILITY VALUE:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/f8cab53a-220d-4398-9512-f272b8b176e3)
## PREDICTION VALUE OF MEAN:
![image](https://github.com/srimathi-25/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/114581999/6ac7a21a-a6ab-4b7f-b1ee-a44a07386551)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
