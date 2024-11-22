# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: 
RegisterNumber:
import pandas as pd
import numpy as np
data=pd.read_csv(r"C:\Users\Suriya\Documents\Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data.drop(['sl_no','salary'],axis=1)
print(data1)
print()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
X=data1.iloc[:,: -1]
Y=data1["status"]
theta=np.random.randn(X.shape[1])
y=Y
def sigmoid(z):
    return 1/(1+np.exp(-z))
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+ (1-y) * np.log(1-h))
def gradient_descent(theta,X,y,alpha,num_iterations):
    m=len(y)
    for i in range(num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y)/m
        theta-=alpha*gradient
    return theta
theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
def predict(theta,X):
    h=sigmoid(X.dot(theta))
    y_pred=np.where(h>=0.5 , 1,0)
    return y_pred
y_pred=predict(theta,X)
accuracy=np.mean(y_pred.flatten()==y) 
print("Accuracy:",accuracy)
print()
print("Predicted:\n",y_pred)
print()
print("Actual:\n",y.values)
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print()
print("Predicted Result:",y_prednew)
 
*/
```

## Output:
    gender  ssc_p    ssc_b  hsc_p    hsc_b     hsc_s  degree_p   degree_t  \
0        M  67.00   Others  91.00   Others  Commerce     58.00   Sci&Tech   
1        M  79.33  Central  78.33   Others   Science     77.48   Sci&Tech   
2        M  65.00  Central  68.00  Central      Arts     64.00  Comm&Mgmt   
3        M  56.00  Central  52.00  Central   Science     52.00   Sci&Tech   
4        M  85.80  Central  73.60  Central  Commerce     73.30  Comm&Mgmt   
..     ...    ...      ...    ...      ...       ...       ...        ...   
210      M  80.60   Others  82.00   Others  Commerce     77.60  Comm&Mgmt   
211      M  58.00   Others  60.00   Others   Science     72.00   Sci&Tech   
212      M  67.00   Others  67.00   Others  Commerce     73.00  Comm&Mgmt   
213      F  74.00   Others  66.00   Others  Commerce     58.00  Comm&Mgmt   
214      M  62.00  Central  58.00   Others   Science     53.00  Comm&Mgmt   

    workex  etest_p specialisation  mba_p      status  
0       No     55.0         Mkt&HR  58.80      Placed  
1      Yes     86.5        Mkt&Fin  66.28      Placed  
2       No     75.0        Mkt&Fin  57.80      Placed  
3       No     66.0         Mkt&HR  59.43  Not Placed  
4       No     96.8        Mkt&Fin  55.50      Placed  
..     ...      ...            ...    ...         ...  
210     No     91.0        Mkt&Fin  74.49      Placed  
211     No     74.0        Mkt&Fin  53.62      Placed  
212    Yes     59.0        Mkt&Fin  69.72      Placed  
213     No     70.0         Mkt&HR  60.23      Placed  
214     No     89.0         Mkt&HR  60.22  Not Placed  

[215 rows x 13 columns]

Accuracy: 0.827906976744186

Predicted:
 [1 1 1 0 1 1 0 1 1 1 0 1 0 1 0 1 0 0 0 0 1 1 0 1 1 0 1 1 1 1 1 1 1 1 0 1 0
 1 1 1 1 0 0 1 1 1 0 1 0 0 1 0 0 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 1 1 1 1 1 1
 0 0 1 1 1 0 1 1 1 1 1 1 1 0 0 1 1 0 1 0 1 1 1 0 1 1 0 1 1 1 1 1 0 1 1 0 1
 0 1 1 1 1 0 1 1 0 0 0 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 0 1 1 1 1 1 0 1 1 1
 1 0 1 1 1 0 0 0 1 1 1 0 1 0 1 0 1 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0 1 0 0 1 0
 1 0 1 0 0 1 1 1 0 0 1 1 1 0 1 1 0 1 0 1 1 0 1 0 1 1 1 0 1 0]

Actual:
 [1 1 1 0 1 0 0 1 1 0 1 1 0 1 0 1 1 0 0 1 1 1 1 1 1 0 1 1 1 0 1 0 1 1 0 1 0
 1 1 1 1 0 0 1 1 0 0 1 1 0 1 0 0 1 1 1 1 1 1 1 1 1 1 0 1 0 1 1 0 1 1 1 1 1
 1 0 1 1 1 0 1 1 0 1 1 1 1 0 1 1 1 0 1 0 1 1 1 0 1 0 0 1 1 1 1 0 0 1 1 0 1
 0 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 0 1 1 1 1 0 1 1 0 1 1 1
 1 0 1 1 1 1 1 0 1 1 0 0 1 0 1 1 1 0 1 0 0 0 0 1 1 0 1 0 1 1 1 0 1 0 0 1 0
 1 0 1 0 0 0 1 1 1 0 1 1 1 0 1 1 0 1 1 1 1 0 1 0 1 1 1 1 1 0]

Predicted Result: [1]


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

