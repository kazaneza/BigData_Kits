import pandas as pd
import numpy as np

My_Data = pd.read_excel("My_Data.xlsx")

# Calculate the mean value of the "QUIZZES" column
mean_quiz = My_Data["QUIZZES"].mean()

# Print the mean value of the "QUIZZES" column
