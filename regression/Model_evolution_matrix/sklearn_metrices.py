from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

#True answers(what actally happened)
y_true=[1,0,1,1,0,0]

#model prediction(what model predicted)
y_pred=[1,1,1,0,1,0]

#evaluation
print("accuracy score",accuracy_score(y_true,y_pred))
print("precision_score",precision_score(y_true,y_pred))
print("recall_score",recall_score(y_true,y_pred))
print("f1_score",f1_score(y_true,y_pred))