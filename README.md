# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the CSV file into a Pandas DataFrame and inspect the data using .head(), .tail(), and .info().
2. Drop unnecessary columns (e.g., 'sl_no') and convert categorical columns to appropriate data types.
Convert categorical variables (e.g., 'gender', 'ssc_b', etc.) into numerical codes using astype('category') and .cat.codes.
Split the data into training and testing sets using train_test_split().
3. Initialize and train a Logistic Regression model on the training data using clf.fit().
4. Predict the target variable for the test set using clf.predict().
Calculate and display the confusion matrix and accuracy score using confusion_matrix() and accuracy_score().

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: NARMADHA SREE S
RegisterNumber:  212223240105
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
*/
```
## READ THE DATA:
```
a=pd.read_csv('Employee (1).csv')
a
```
# output:
![image](https://github.com/user-attachments/assets/31afa714-8014-4fed-b91b-b12dcb5caf1d)


# Info :
```
a.head()
a.tail()
a.info()
```
# output:
![image](https://github.com/user-attachments/assets/fa4ff54b-4640-4b4b-8ace-865685338f4e)
![image](https://github.com/user-attachments/assets/cbd5ba0c-f122-4209-ba03-a07009d59573)
![image](https://github.com/user-attachments/assets/6e95e2ce-ecf6-47be-b7d3-d2d94acb4f36)
## Dataset transformed head:
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

a["salary"]=le.fit_transform(data["salary"])
a.head()
```
# output:
![image](https://github.com/user-attachments/assets/ffe7295d-5b59-4267-8ee6-66aea79234c6)
# x.head():
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
x.head()
```
# output:
![image](https://github.com/user-attachments/assets/42bf8aad-82d4-4825-9a3a-47601ac5cef0)
# Accuracy:
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![image](https://github.com/user-attachments/assets/48ff8705-42e8-49b8-9b2b-3a5a81fc210c)
# predicted:
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
# output:
![image](https://github.com/user-attachments/assets/9be9a1c0-64f8-4fea-815d-1151b4e2a8c8)
# RESULT:
Thus the program to implement the Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
