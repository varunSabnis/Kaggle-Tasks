


import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
import numpy as np
import zipfile






df = pd.read_csv('..\data\creditcard.csv')




print('df',df)
print('df shape',df.shape)




from sklearn.preprocessing import StandardScaler 
df['normAmount'] = StandardScaler().fit_transform(df['Amount'].reshape(-1, 1))
df = df.drop(['Time','Amount'],axis=1)
print('new df shape',df.shape)




X = df.loc[:,df.columns!='Class']
Y = df.loc[:,df.columns=='Class']
print('shape of X',X.shape)
print('shape of Y',Y.shape)




#get X_train_undersampled , Y_train_undersampled 
#let number of negative examples be same as number of positive examples
#there are 492 positive samples, so randomly pick 492 negative sample
fraud_count = len(df.loc[df['Class']==1])
fraud_indices = np.array(df.loc[df['Class']==1].index)

normal_indices = df.loc[df['Class']==0].index
np.random.seed(5)
random_normal_indices = np.random.choice(normal_indices,fraud_count,replace=False)
random_normal_indices = np.array(random_normal_indices)

final_indices = np.concatenate([fraud_indices,random_normal_indices])
print('final indices for training',final_indices)





X_undersampled = df.iloc[final_indices,:]
print(type(X_undersampled))
X_undersampled = X_undersampled.loc[:,X_undersampled.columns!='Class']
print(type(X_undersampled))
Y_undersampled = df.iloc[final_indices,:]
print(type(Y_undersampled))
Y_undersampled = Y_undersampled.loc[:,Y_undersampled.columns=='Class']
print(type(Y_undersampled))





print(X_undersampled.shape)
print(Y_undersampled.shape)





from sklearn.cross_validation import train_test_split
X_undersampled_train, X_undersampled_test, Y_undersampled_train, Y_undersampled_test = train_test_split(X_undersampled,Y_undersampled,test_size = 0.3, random_state = 0)


:


from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report





def KFoldCrossValidation(X_undersampled_train,Y_undersampled_train):
    Fold = KFold(len(Y_undersampled_train),5,shuffle=False,random_state=0)
    print(Fold)
    c_param_range = [0.01,0.1,1,10,100]
    recall_averages = []
    c_best = 0
    max_recall_average = 0
    for c_param in c_param_range:
        recall_acc_average = 0
        rec_accuracy=0
        for iteration,indices in enumerate(Fold,start=1):
            lr=LogisticRegression(C=c_param,penalty='l1')
            lr.fit(X_undersampled_train.iloc[indices[0],:],Y_undersampled_train.iloc[indices[0],:].values.ravel())
            Y_undersampled_prediction = lr.predict(X_undersampled.iloc[indices[1],:].values)
            rec_accuracy = recall_score(Y_undersampled_train.iloc[indices[1],:].values,Y_undersampled_prediction)
            recall_acc_average = recall_acc_average + rec_accuracy
        recall_acc_average = recall_acc_average / 5
        if(recall_acc_average>=max_recall_average):
            max_recall_average = recall_acc_average
            c_best = c_param
        recall_averages.append(recall_acc_average)
    return(c_best,recall_averages)
    





c_best,recall_averages = KFoldCrossValidation(X_undersampled_train,Y_undersampled_train)
print('c_best',c_best)
print('recall_averages',recall_averages)





#c best value is 0.01, c is the learning rate parameter
lr = LogisticRegression(C=c_best,penalty='l1')
lr.fit(X_undersampled_train,Y_undersampled_train.values.ravel())
Prediction = lr.predict(X_undersampled_test.values)
rec_accuracy = recall_score(Y_undersampled_test.values,Prediction)
print('Recall Accuracy', rec_accuracy)





#Plotting ROC curve
lr = LogisticRegression(C = c_best, penalty = 'l1')
Y_pred_undersample_score = lr.fit(X_undersampled_train,Y_undersampled_train.values.ravel()).decision_function(X_undersampled_test.values)
#Y_pred_undersample_score = Y_pred_undersample_score[:,0]
print(np.round(Y_pred_undersample_score,2))
print(Y_pred_undersample_score.shape)





fpr, tpr, thresholds = roc_curve(Y_undersampled_test.values.ravel(),Y_pred_undersample_score)
print('length of false positive rate',len(fpr))
print('length of true positive rate',len(tpr))
print('thresholds',np.round(thresholds,2))
roc_auc = auc(fpr,tpr)
print('roc_auc',roc_auc)




plt.title('ROC curve')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
#plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')

plt.show()




#For revision and trying out other techniques
#Try oversampling
#Try clustering **
#Try SMOTE ** 

