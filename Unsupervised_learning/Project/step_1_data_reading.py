import pandas as pd

df=pd.read_csv(r"D:\Machine-learning-using-scikit-learn\Unsupervised_learning\Project\student_data_dataset.csv")

print("sample rows")
print(df.head())

print('Dataset shape')
print(f'Rows: {df.shape[0]}, columns: {df.shape[1]}')

print('dataset info')
print(df.info())

print('Dataset summary')
print(df.describe(include='all'))

print('missing values')
print(df.isnull().sum())
#now you havr read the data now have to process it 
#so see if have any missing values or having text values lkke non-numeric if there are convert into numeric using one-hot encoding or getdumies method