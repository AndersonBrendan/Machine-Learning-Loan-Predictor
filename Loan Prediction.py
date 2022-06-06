import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib as mplt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

######    PreProcessing   ######

Dataframe = pd.read_csv("Loan Train Dataset.csv")
Test = pd.read_csv("Loan Test Dataset.csv")

Dataframe['Loan_Status'] = Dataframe.Loan_Status.map({'Y': 1, 'N': 0}).astype(int)

Dataframe = Dataframe.replace({"Gender":{"Male": 1, "Female":0}})
Dataframe = Dataframe.replace({"Married": {"Yes":1 , "No":0}})
Dataframe["Dependents"] = Dataframe["Dependents"].replace('3+', '3')
Dataframe["Dependents"]= pd.to_numeric(Dataframe["Dependents"], errors='coerce')

Dataframe['Self_Employed'].value_counts()
Dataframe= Dataframe.replace({"Self_Employed":{"Yes":1, "No":0 }})

Dataframe['Education'].value_counts()
Dataframe= Dataframe.replace({"Education":{"Graduate":1, "Not Graduate":0 }})

Dataframe = Dataframe.drop(columns=['Loan_ID'])

Dataframe['Property_Area'].value_counts()
Dataframe['Property_Area'] = Dataframe['Property_Area'].map({'Rural': 0, 'Urban': 1, 'Semiurban': 2})

print(Dataframe)


#Fil the NaN values
Dataframe.fillna(Dataframe.median(), inplace=True)
columns = Dataframe.columns
for column in columns:
  Dataframe[column] = pd.to_numeric(Dataframe[column], errors='coerce')


#Check for missing data
#col_names= Dataframe.columns.tolist()
#for column in col_names:
# print("Null and <{0}>: <{1}>".format(column,Dataframe[column].isnull().sum()))


### Classifier ###


x = Dataframe.iloc[:,:-1].values
y = Dataframe.iloc[:,-1].values

scaler = MinMaxScaler()
X = scaler.fit_transform(x)

#split our training and testing data
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=0)



SVC_classifier = SVC(kernel = 'rbf', gamma= 0.2)
SVC_classifier.fit(x_train, y_train)

y_predict = SVC_classifier.predict(x_test)

c_matrix = confusion_matrix(y_test, y_predict)
print(c_matrix)


accuracies = cross_val_score(estimator = SVC_classifier, X = x_train, y = y_train, cv = 10)
print("Model Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation is: {:.2f} %".format(accuracies.std()*100))