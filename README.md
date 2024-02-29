# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and read the dataframe.

2.Assign hours to X and scores to Y.

3.Implement training set and test set of the dataframe

4.Plot the required graph both for test data and training data.

5.Find the values of MSE , MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SASIDHARAN P
RegisterNumber:  212223080051
=/
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='blue')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```
## Output:
Dataset

![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/ae400274-da32-4dd0-8bb3-3cfe763ca8e2)

df.head()


![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/3dc55ec3-d417-422d-a4ce-01163e5fa65f)

f.tail()

![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/fdebe179-51c6-4f37-a547-09926b0f889c)

X and Y values

![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/35b029b4-a021-4487-bff9-59a596136f8a)

Predication values of X and Y

![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/c486664b-436d-4ba3-9ab0-f5bbaf4602cd)

MSE,MAE and RMSE

![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/b3c93a03-520f-4d9c-9799-4d7a5c72ff84)

Training Set


![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/74fb92a6-2de7-4d0b-837f-aeb6e5c03117)

Testing Set

![image](https://github.com/sasirath13/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/160568449/708ce833-d038-41ad-8904-fc6e1ffaa93d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
