import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'D:\Machine-learning-using-scikit-learn\Unsupervised_learning\Project\student_data_dataset.csv')

#2.data processing
le=LabelEncoder()
df['Internet']=le.fit_transform(df['Internet'])
df['Passed']=le.fit_transform(df['Passed'])

# 3.feature scaling
features=['StudyHours','Attendance','PastScore','SleepHours']
scalar=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scalar.fit_transform(df[features])

# 4.split the data
X=df_scaled[features]#features
y=df_scaled['Passed']#target        because we want to check fail or pass
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#5.select the model
model=LogisticRegression()#as due to binary otput fil or pas using this model
model.fit(X_train,y_train)

# 6.making predictions
y_pred=model.predict(X_test)

# 7.evaluating model 
print("classificatio report")
print(classification_report(y_test,y_pred))

conf_matrix=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["Fail","Pass"],yticklabels=["Fail","Pass"])
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix")
plt.tight_layout()
plt.show()


# use user input to predict their results
print("-------Predict your result-------")
try:
    study_hours=float(input("Enter study hours:"))
    attendence=float(input("Enter attendence:"))
    past_score=float(input("Enter past score:"))
    sleep_hours=float(input("Enter sleep hours:"))

    user_input_df=pd.DataFrame([{
        'StudyHours':study_hours,
        'Attendance': attendence,
        'PastScore':past_score,
        'SleepHours':sleep_hours
    }])

   
      #now scale the user inputs
    user_input_scaled=scalar.transform(user_input_df)

    prediction=model.predict(user_input_scaled)[0]
    

    result="Pass" if prediction == 1 else "Fail"
    print(f"Prediction Based on input: {result}")
except Exception as e:
    print("an error occured",e)    
    