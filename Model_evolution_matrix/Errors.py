from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error
import numpy as np

real_scores=[90,85,95,75,80]
predicted_scores=[90,80,90,70,85]

mae=mean_absolute_error(real_scores,predicted_scores)
mse=mean_squared_error(real_scores,predicted_scores)
rmse=root_mean_squared_error(real_scores,predicted_scores)#down line can also be used for rmse
remse=np.sqrt(mse)

print("mean_absolute_error:on average off by",mae)
print("mean_squared_error:squared mistake values",mse)
print("root_mean_squared_error:final realistic error",rmse)
print("root_mean_squared_error",remse)
