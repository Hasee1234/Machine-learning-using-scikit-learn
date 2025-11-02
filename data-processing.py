import pandas as pd
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv('big-black-money.csv')
# print(df.describe())
print(df.isnull().sum())


