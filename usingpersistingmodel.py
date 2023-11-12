import pandas as pd
from sklearn import svm
import joblib

#user interface
quiz=int(input("Enter Quiz Marks: "))
Assign=int(input("Enter Assignment Marks: "))
Mid=int(input("Enter Mid-Semester Marks: "))
Final=int(input("Enter Final-Semester Marks: "))

#load joblib

model=joblib.load("svm_grade_recommender.joblib")

print("Grade: ",model.predict([[quiz,Assign,Mid,Final]]))