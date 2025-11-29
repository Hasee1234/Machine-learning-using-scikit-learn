#pca used to educe dimentionality
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler#to scale data
from sklearn.decomposition import PCA#pca for decomposition break the data into features and then select the necessary features

data={
    'Age':[25,30,35,40,45,50],
    'Income':[30000,35000,38000,40000,42000,45000],
    'Spending':[10,20,40,60,30,50],
    'Savings':[1000,5000,3000,2000,6000,4000]
}
df=pd.DataFrame(data)

# now standerize data 
scaler=StandardScaler()
#fit_transform means fit:learn from mean and standard deviation data and transform:now use the logic to scale the data          now the data is fully structured
scaled_data=scaler.fit_transform(df)

pca=PCA(n_components=2)#n_components means how many features you want to keep and they should important
pca_result=pca.fit_transform(scaled_data)

pca_df=pd.DataFrame(pca_result,columns=['PCA1','PCA2'])#these two columns will have compressed data

explained_variance=pca.explained_variance_ratio_#it will tell how much percent of information is captured on every component
print('variance captured by each PCA component:')
print(np.round(explained_variance * 100 , 2))

plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'],color='black',s=80)
plt.title('PCA projection (2D view)')
plt.xlabel('PCA1 main pattern')
plt.ylabel('PCA2 minor pattern')
plt.grid(True)
plt.show()

print("new data with PCA features PCA1 PCA2")
print(pca_df)