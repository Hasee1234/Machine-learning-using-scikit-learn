from sklearn.linear_model import LogisticRegression

X=[[1],[2],[3],[4],[5]]#hours studied
y=[40,50,60,70,80]#result 0=fail,1=pass

model=LogisticRegression()
model.fit(X,y)
hours=float(input("Enter how many hours you study"))
result=model.predict([[hours]])[0]#0 the answer is in list we cant show answer in list so to show answer we write [0]width 0
if result == 1:
    print(f"Based on your study you are likely to pass")
else:    
    print(f"Based on your study you are likely to fail")