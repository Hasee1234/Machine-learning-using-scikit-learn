import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#sample data
data={
    'Name':['a','b','c','d','e','f'],
    'Age':[20,25,30,23,35,40],
    'spending':[400,500,300,250,100,1000]
}
df=pd.DataFrame(data)

X=df[['Age','spending']]
model=KMeans(n_clusters=2,random_state=42,n_init=10)
# this model will group your data
# n_cluster: will tell in how many groups you want to divide 
# random_state=42:it will always start from random point or centroid   42 is a safe number so used this
#n_init=10:will do random selection 10 times    ,this is good for small dtasets you will aslo use more tha  10 in large datasets

df['groups']=model.fit_predict(X)
# fit_predict means fit:learn from data,predict:now predict

plt.figure(figsize=(10,6))
for group in df['groups'].unique():#this line means you have to do something to all people you find    .unique gives distant values
    group_data=df[df['groups'] == group]#it will filter data called masking
    plt.scatter(group_data['Age'],group_data['spending'],label=f'Group {group}')

plt.xlabel('Age')
plt.ylabel('Spending')
plt.title('Customers Segments (k_means)')
plt.legend()
plt.grid(True)
plt.show()

print(df)

#SO USING THE GRAPH you can group people in premium and normal buyers