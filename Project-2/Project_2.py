import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

data=pd.read_csv(r'D:\Machine-learning-using-scikit-learn\Project-2\student_performance.csv')
X=data[['weekly_self_study_hours']]
y=data['total_score']
model=LinearRegression()
model.fit(X,y)
predicted_score=model.predict(X)

mae=mean_absolute_error(y,predicted_score)
mse=mean_squared_error(y,predicted_score)
rmse=np.sqrt(mae)
r2=r2_score(y,predicted_score)#closer to 1 ibetter

print('mean_absolute_error',round(mae,2))
print('mean_squared_error',round(mse,2))
print('mean_squared_error',round(rmse,2))
print('r2_score(Model accuracy)',round(r2,2))

new_hour=9
predicted__new_score=model.predict([[new_hour]])
print(f'based on your {new_hour} study hours you will get {predicted__new_score} scores')

plt.figure(figsize=(10,6))
plt.hist(data['total_score'],bins=30,color='green',edgecolor='black')
plt.title("distributon of total score")
plt.ylim(0,100000)
plt.xlabel('final exam scores')
plt.ylabel('Number of students')
plt.grid(True)
plt.show()
#if plot has scores evenly spread like all bars of about same hight so we cant distiguish toppers then we draw scatter plot + regression line


plt.figure(figsize=(10,6))
plt.scatter(X,y,color='blue',label='actual scores')
plt.plot(X,predicted_score,color='red',label='predicted scores(regression line)')
plt.title('Model prediction vs actual scores')
plt.xlabel('study hours per week')
plt.ylabel('final output')
plt.grid(True)
plt.show()

