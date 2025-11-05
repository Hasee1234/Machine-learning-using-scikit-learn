from sklearn.tree import DecisionTreeClassifier
X=[
    [3,4],#size and shade
    [6,7],
    [8,9],
    [9,10],
]
y=[0,0,1,1]#0=apple,1=orange

model=DecisionTreeClassifier()
model.fit(X,y)
size=float(input("Enter the size of fruit in cm : "))
shade=float(input("Enter the shade of fruit from 1 to 10 : "))

result=model.predict([[size,shade]])[0]
if result == 0:
    print("your fruit is more likely an apple")
else:    
    print("your fruit is more likely an orange")