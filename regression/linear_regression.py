from sklearn.linear_model import LinearRegression
X=[[1],[2],[3],[4],[5]]
y=[[40],[50],[60],[70],[80]]

model=LinearRegression()
model.fit(X,y)#fit is used to train model
hours=float(input("how many hours did you studied"))
predicted_marks=model.predict([[hours]])

print(f"based on your hours{hours}you may secure {predicted_marks}")