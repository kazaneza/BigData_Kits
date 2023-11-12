#Import all the needed libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import dataset
df=pd.read_csv('malaria_clinical_data_update2.0.csv')
#df.dropna(inplace = True)
#split dataset into training and test set
X=df.drop(columns=['Unnamed: 0','SampleID','consent_given','location','bednet','fever_symptom','Suspected_Organism','Suspected_infection','RDT','Microscopy','Laboratory_Results','Clinical_Diagnosis','Blood_culture','Urine_culture','Taq_man_PCR'])
y=df['Clinical_Diagnosis']
X_train,x_test,y_train,y_test=train_test_split (X,y,test_size=0.2)
#create a DecisionTree, LogisticRegression, supportVectorMachine and RandomForestClassifiers
Decision_tree_model=DecisionTreeClassifier()
Logistic_regression_Model=LogisticRegression(solver='lbfgs',max_iter=100000)
SVM_model=svm.SVC(kernel='linear')
RF_model=RandomForestClassifier(n_estimators=100)
#train the models using the training sets
Decision_tree_model.fit(X_train,y_train)
Logistic_regression_Model.fit(X_train,y_train)
SVM_model.fit(X_train,y_train)
RF_model.fit(X_train,y_train)
#predict the model
DT_Prediction=Decision_tree_model.predict(x_test)
LR_Prediction=Logistic_regression_Model.predict(x_test)
SVM_Prediction=SVM_model.predict(x_test)
RF_Prediction=RF_model.predict(x_test)
#calculation of Model accuracy
DT_score=accuracy_score(y_test,DT_Prediction)
LR_score=accuracy_score(y_test,LR_Prediction)
SVM_score=accuracy_score(y_test,SVM_Prediction)
RF_score=accuracy_score(y_test,RF_Prediction)
#display accuracy
print("Decision Tree accuracy=",DT_score*100,"%")
print("Logistic Regression accuracy=",LR_score*100,"%")
print("Support Vector Machine accuracy=",SVM_score*100,"%")
print("Random Forest accuracy=",RF_score*100,"%")
print(df.to_string())