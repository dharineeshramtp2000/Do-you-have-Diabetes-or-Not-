#Importing the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importing the dataset
Dataset = pd.read_csv("pima_native_american_diabetes_weka_dataset.csv")
X = Dataset.iloc[:,:-1]
y = Dataset.iloc[:,-1]

#Lets Feature Scale the independent variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting the dataset into training and testing and performing a good shuffle of data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle = True)

#Now lets define our Decision Tree regression Model.
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion="entropy",max_depth=5)
classifier.fit(X_train, y_train)

#Now lets predict with our test set
y_pred = classifier.predict(X_test)

#Its time for evaluating our model.
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
acc_train = accuracy_score(y_train, classifier.predict(X_train))
f1_train = f1_score(y_train, classifier.predict(X_train), average= 'weighted')

print("Traing set results")
print("ACCURACY ---------------------->",acc_train)
print("F1 SCORE ---------------------->",f1_train)

#Now lets see how well is our model. So now lets evaluate with our test set
acc_test = accuracy_score(y_test, y_pred)
f1_test = f1_score(y_test, y_pred, average= 'weighted')

print("Test set results")
print("ACCURACY ---------------------->",acc_test)
print("F1 SCORE ---------------------->",f1_test)

#Now lets have our famous Confusion Matrix to visually understand.
cm = confusion_matrix(y_test,y_pred)
print(cm)
