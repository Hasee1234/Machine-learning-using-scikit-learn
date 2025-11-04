
# feature scaling
# when you want to convert values in between 0 and 1 to avoid -ve values
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split#to split data into training and testing datset

data={
    'studyHours':[1,2,3,4,5],
    'TestScore':[40,50,60,70,80]
}

df=pd.DataFrame(data)

# scalar=StandardScaler()#mean=0 and sd=1
# X_scaled=scalar.fit_transform()

# scalar=MinMaxScaler()#to limit all values bw 0 and 1
# X_scaled=scalar.fit_transform()


#EXAMPLES
Standard_Scaler=StandardScaler()
Standard_Scaled=Standard_Scaler.fit_transform(df)
print("standard scaler")
print(pd.DataFrame(Standard_Scaled, columns=['studyHours','TestScore'])) 

minmax_scalar=MinMaxScaler()
minmax_scaled=minmax_scalar.fit_transform(df)
print("\nminmax_scaled values",)
print(pd.
DataFrame(minmax_scaled,columns=['studyHours','TestScore']))

#train_test split
X=df[['studyHours']]#double bracket to show dataframe
y=df[['TestScore']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)#random state means you will always get same consistency
print('Training Data')
print(X_train)
print('Testing Data')
print(X_test)
                #used to compare data of X and y
print('Training Data')
print(y_train)
print('Testing Data')
print(y_test)