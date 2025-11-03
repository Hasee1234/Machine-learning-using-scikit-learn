import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('big-black-money.csv')
# print(df.describe())
# print(df.isnull().sum())
df_label=df.copy()#copy to make copy of original data so original data dont change


#label encoding
#  to convert conditions into mathematical form like true false to 0 and 1 
#use for columns havimg more than 2 different values

le=LabelEncoder()
df_label['Report']=le.fit_transform(df_label['Reported by Authority'])#fit transform means learn from the column and change into mathematical form 0 and 1 like here for true false
#report is column we made that will have the mathematical form of Reported by Authority column from data

print(df_label[['Reported by Authority','Report']].head())



#one-hot coding
#use for columns havimg more than 2 different values
#will convert values in differnt columns like [1,0,0,0]
df_encoded=pd.get_dummies(df_label,columns=['Country'])#getdummies to convert values n binary
print("hot coded data\n",df_encoded.head())


