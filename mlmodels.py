import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Step 1: Read the CSV file
Grade = pd.read_csv('malaria_clinical_data.csv')

# Step 2: Prepare the data
x = Grade.drop(columns=['SampleID','consent_given','location','Enrollment_Year', 'bednet', 'fever_symptom',
                        'Suspected_Organism', 'Suspected_infection', 'RDT', 'Blood_culture', 'Urine_culture',
                        'Taq_man_PCR', 'Microscopy', 'Laboratory_Results', 'Clinical_Diagnosis', 'rbc_count',
                        'RBC_dist_width_Percent'])
y = Grade['Clinical_Diagnosis']

# Step 3: Handle missing values
imputer = SimpleImputer(strategy='mean')
x = pd.DataFrame(imputer.fit_transform(x), columns=x.columns)

# Step 4: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)  # Adjust test_size value

# Step 5: Initialize and train the models
Decision_Tree_Model = DecisionTreeClassifier()
Logistic_Regression_Model = LogisticRegression(solver='lbfgs', max_iter=1000)
svm_model = svm.SVC(kernel='linear')
Random_Forest_model = RandomForestClassifier(n_estimators=100)

Decision_Tree_Model.fit(x_train, y_train)
Logistic_Regression_Model.fit(x_train, y_train)
svm_model.fit(x_train, y_train)
Random_Forest_model.fit(x_train, y_train)

# Step 6: Evaluate the models
print("Decision Tree accuracy:", accuracy_score(y_test, Decision_Tree_Model.predict(x_test)) * 100, "%")
print("Logistic Regression accuracy:", accuracy_score(y_test, Logistic_Regression_Model.predict(x_test)) * 100, "%")
print("SVM accuracy:", accuracy_score(y_test, svm_model.predict(x_test)) * 100, "%")
print("Random Forest accuracy:", accuracy_score(y_test, Random_Forest_model.predict(x_test)) * 100, "%")
