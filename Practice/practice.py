# Load a dataset and explore its structure: shape, head, summary, and apply filters
import pandas as pd
# data=pd.read_csv('big-black-money.csv')
# print(data.shape())
# print(data.head())
# print(data.describe())


# from sklearn.linear_model import LinearRegression
# X=[[1],[2],[3],[4],[5]]#marlas 
# y = [100, 200, 300, 400, 500]  # price in Rs

# model=LinearRegression()
# model.fit(X,y)
# size=float(input("Enter the size of house you want to buy"))
# price=model.predict([[size]])
# print(f"Based on your required {size} marlas house size ,the price will be {price}")

from sklearn.linear_model import LogisticRegression
X=[[1],[2],[3],[4],[5]]
y=[0,0,0,1,1]

model=LogisticRegression()
model.fit(X,y)
hours=float(input("Enter the number of yours you studied"))
result=model.predict([[hours]])[0]
if result == 1:
    print("Based on your studyhours you are likely to pass")
else:    
    print("Based on your studyhours you are likely to fail")

print(result)