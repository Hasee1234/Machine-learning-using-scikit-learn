from sklearn.neighbors import KNeighborsClassifier
X=[
    [200,6],
    [250,6.5],
    [300,7],
    [320,8],
    [340,8.5],
    [400,9],
]
#apple=0 , orange=1
y=[0,0,0,1,1,1]

model=KNeighborsClassifier()
model.fit(X,y)
weight=float(input("enter the weight of fruit you have in grams :"))
size=float(input("enter the size of fruit you have in cm :"))

prediction=model.predict([[weight,size]])[0]
if prediction == 0:
    print("this is likely an apple")
else:    
    print("this is likely an orange")