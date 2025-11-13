from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error,mean_absolute_error,mean_squared_error
import numpy as np
import pandas as pd

Data=pd.read_csv(r"D:\Machine-learning-using-scikit-learn\Project-1\student.csv")

X=Data[["hours"]]
# y=Data[["report"]]
y=Data["report"]

model=LinearRegression()
model.fit(X,y)
predicted_score=model.predict(X)

# print('predicted score : ',predicted_score)

mae=mean_absolute_error(y,predicted_score)
mse=mean_squared_error(y,predicted_score)
rmse=root_mean_squared_error(y,predicted_score)

print("mean_squared_error : ",mae)
print("mean_absolute_error : ",mse)
print("root_mean_squared_error : ",rmse)
print("root_mean_squared_error : ",np.sqrt(mae))



input_hours=float(input("enter the number of hours you studied : "))
new_predicted_score=model.predict(pd.DataFrame([[input_hours]],columns=["hours"]))
# new_predicted_score=model.predict([[input_hours]])
print(f"based on your study you will get {new_predicted_score} marks")



# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import root_mean_squared_error,mean_absolute_error,mean_squared_error
# import numpy as np
# import pandas as pd

# Data=pd.read_csv(r"D:\Machine-learning-using-scikit-learn\Project-1\student.csv")

# X=Data[['hours']]#2d
# y=Data['report']

# model=LinearRegression()
# model.fit(X,y)

# predicted_score = model.predict(X)

# mae=mean_absolute_error(y,predicted_score)
# mse=mean_squared_error(y,predicted_score)
# rmse=np.sqrt(mae)

# print(mae)
# print(mse)
# print(rmse)

# new_hour = float(input("enter an hour : "))
# new_pred=model.predict([[new_hour]])
# print(new_pred)