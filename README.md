# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import pandas

2.Import Decision tree classifier

3.Fit the data in the model

4.Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Harshitha D
RegisterNumber:212224040110  
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
```
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
```
```
x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
```
```
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn.metrics import r2_score
r2=r2_score(y_test,y_pred)
print("R2 Score = ",r2)
```
```
dt.predict([[5,6]])
```
```
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(12,8))
plot_tree(
    dt,
    feature_names=["Position", "Level"],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for Salary Prediction", fontsize=14)
plt.show()
```






## Output:

<img width="400" height="436" alt="500887975-5276cf09-2dd1-452a-9ed1-dd0bf4338bce" src="https://github.com/user-attachments/assets/61791311-074d-4678-8528-19e90dc1c7a4" />

<img width="311" height="295" alt="500888057-2a9f2d89-dad9-475c-8c48-299aba05c0d4" src="https://github.com/user-attachments/assets/0764d7a9-c565-4a32-8b6f-ac37c78ed8a5" />

<img width="315" height="25" alt="500888134-fc42b67e-1a52-46bf-b144-51df3a83d9bb" src="https://github.com/user-attachments/assets/04e32a84-47c0-4a5a-ae0e-bd866d6ee971" />

<img width="1270" height="850" alt="500888188-53c66b62-a5c0-4086-abd7-c3e91abd7cb6" src="https://github.com/user-attachments/assets/ad4a0e58-0341-4f12-9365-5d64b2af5a77" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
