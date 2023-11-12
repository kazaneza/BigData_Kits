import pandas as pd
from sklearn.tree import  DecisionTreeClassifier

Grade = pd.read_excel('Data.xlsx')
x=Grade.drop(columns=['SNAMES ','Total Marks','Marks /20','Grading '])
y=Grade['Grading ']
model=DecisionTreeClassifier()
model.fit(x.values,y)
prediction = model.predict([[12,12,27,18]])
print(prediction)

