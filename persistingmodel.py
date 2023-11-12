import pandas as pd
from sklearn import svm
import joblib


grade_data=pd.read_csv('cleanData.csv')
x=grade_data.drop(columns=['SampleID','consent_given','location','Enrollment_Year', 'bednet', 'fever_symptom',
                        'Suspected_Organism', 'Suspected_infection', 'RDT', 'Blood_culture', 'Urine_culture',
                        'Taq_man_PCR', 'Microscopy', 'Laboratory_Results', 'Clinical_Diagnosis', 'rbc_count',
                        'RBC_dist_width_Percent'])
y=grade_data['Clinical_Diagnosis']
svm_model=svm.SVC(kernel='linear')
svm_model.fit(x.values,y)

joblib.dump(svm_model,'svm_malaria_recommender.joblib')