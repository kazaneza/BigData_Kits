import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data=pd.read_csv('malaria_clinical_data.csv')

data["temperature"].fillna(data["temperature"].median(),inplace=True)
data["parasite_density"].fillna(data["parasite_density"].median(),inplace=True)
data["temperature"].fillna(data["temperature"].median(),inplace=True)
data["wbc_count"].fillna(data["wbc_count"].median(),inplace=True)
data["rbc_count"].fillna(data["rbc_count"].median(),inplace=True)
data["hb_level"].fillna(data["hb_level"].median(),inplace=True)
data["hematocrit"].fillna(data["hematocrit"].median(),inplace=True)
data["mean_cell_volume"].fillna(data["mean_cell_volume"].median(),inplace=True)
data["mean_corp_hb"].fillna(data["mean_corp_hb"].median(),inplace=True)
data["mean_cell_hb_conc"].fillna(data["mean_cell_hb_conc"].median(),inplace=True)
data["platelet_count"].fillna(data["platelet_count"].median(),inplace=True)
data["platelet_distr_width"].fillna(data["platelet_distr_width"].median(),inplace=True)
data["mean_platelet_vl"].fillna(data["mean_platelet_vl"].median(),inplace=True)
data["neutrophils_percent"].fillna(data["neutrophils_percent"].median(),inplace=True)
data["lymphocytes_percent"].fillna(data["lymphocytes_percent"].median(),inplace=True)
data["mixed_cells_percent"].fillna(data["mixed_cells_percent"].median(),inplace=True)
data["neutrophils_count"].fillna(data["neutrophils_count"].median(),inplace=True)
data["lymphocytes_count"].fillna(data["lymphocytes_count"].median(),inplace=True)
data["mixed_cells_count"].fillna(data["mixed_cells_count"].median(),inplace=True)
data["RBC_dist_width_Percent"].fillna(data["RBC_dist_width_Percent"].median(),inplace=True)


print(data.duplicated())
data.drop_duplicates(inplace=True)

print(data)

X=data.drop(columns=["SampleID","consent_given","location","Enrollment_Year","bednet","fever_symptom","Suspected_Organism","Suspected_infection","RDT","Blood_culture","Urine_culture","Taq_man_PCR","Microscopy","Laboratory_Results","Clinical_Diagnosis","rbc_count","RBC_dist_width_Percent"])

Y=data["Clinical_Diagnosis"]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2)

dtc=DecisionTreeClassifier()
svm=svm.SVC(kernel="linear")
lr=LogisticRegression(solver="lbfgs",max_iter=1000)
rfc=RandomForestClassifier(n_estimators=100)

dtc.fit(x_train,y_train)
svm.fit(x_train,y_train)
lr.fit(x_train,y_train)
rfc.fit(x_train,y_train)

print("DecisionTreeClassifier: ",accuracy_score(y_test, dtc.predict(x_test))*100,"%")
print("SVM: ",accuracy_score(y_test, svm.predict(x_test))*100,"%")
print("LogisticRegression: ",accuracy_score(y_test, lr.predict(x_test))*100,"%")
print("RandomForestClassifier: ",accuracy_score(y_test, rfc.predict(x_test))*100,"%")


joblib.dump(svm,"malaria_model.joblib")

