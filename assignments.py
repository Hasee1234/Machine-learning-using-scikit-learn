import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,LogisticRegression

# data=pd.read_csv(r'D:\Machine-learning-using-scikit-learn\big-black-money.csv')
# # print(data.isnull().head())
# # label encoding 
# le=LabelEncoder()
# data['report']=le.fit_transform(data['Reported by Authority'])
# print(data['report'].head())

# #one hot encoding
# ohe=pd.get_dummies(data["Transaction Type"])
# print(ohe.head())


# # feature_scaling 
# dataset={
#     'studyHours':[1,2,3,4,5],
#     'TestScore':[40,50,60,70,80]
# }
# df=pd.DataFrame(dataset)
# standard_scalar=StandardScaler()
# standard_scaled_data=standard_scalar.fit_transform(df)
# print(pd.DataFrame(standard_scaled_data,columns=['studyHours','TestScore']))

# minmax_scaler=MinMaxScaler()
# minmax_scaled_data=minmax_scaler.fit_transform(df)
# print(pd.DataFrame(minmax_scaled_data,columns=['studyHours','TestScore']))


# X=dataset['studyHours']
# y=dataset['TestScore']
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
# print(X_test,X_train,y_test,y_train)


# data={
#     'size':[[5],[6],[7],[8],[9]],
#     # 'price':[[500000],[600000],[700000],[800000],[900000]]
#     'price':[500000,600000,700000,800000,900000]
# }
# X=data['size']
# y=data['price']
# model=LinearRegression()
# model.fit(X,y)
# new_size=float(input('enter the size of house in one marlas'))
# new_price=model.predict([[new_size]])[0]
# print(f'you will get your required {new_size} marla size house in {new_price} rupees')


data={
    'study_Hours':[[1],[2],[3],[4],[5]],
    'score':[60,70,80,90,100]
}
X=data['study_Hours']
y=data['score']

model=LogisticRegression()
model.fit(X,y)
your_time=float(input("Enter your study hours"))
your_score=model.predict([[your_time]])[0]
if your_score == 1:
    print(f"based on your study hours:{your_time},you will might pass")
else:    
    print(f"based on your study hours:{your_time},you will might fail")

print(your_score)