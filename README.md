# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

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
Program to implement the prediction of iris species using SGD Classifier.
Developed by: NIJANTH K
RegisterNumber: 212224032186 
*/
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
iris=load_iris()
df=pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target
print(df.head())
X=df.drop('target',axis=1)
y=df['target']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
sgd_clf=SGDClassifier(max_iter=1000,tol=1e-3)
sgd_clf.fit(X_train,y_train)
y_pred=sgd_clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(f"Acuuracy:{accuracy:.3f}")
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(cm)

```

## Output:
![prediction of iris species using SGD Classifier](sam.png)
![image](https://github.com/user-attachments/assets/b8bdcef2-ff2d-4265-a496-96ced1977680)
Y_PRED
![image](https://github.com/user-attachments/assets/f1579470-7f1e-44e0-8814-6591f5166591)

ACCURACY
![image](https://github.com/user-attachments/assets/58c78296-0f27-47a5-88c7-058e24261037)

CONFUSION MATRIX
![image](https://github.com/user-attachments/assets/9573e293-eb12-4c13-bc9a-9a69f8ebdcb1)

## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
