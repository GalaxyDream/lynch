from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from preprocessing import load_trainingdata
from preprocessing import load_testingdata
import cPickle
from sklearn import metrics

X_train,y_train=load_trainingdata()
rf_model=AdaBoostClassifier(RandomForestClassifier(n_estimators=1000,max_features=50,criterion="entropy",max_depth=4,oob_score=True,verbose=1))
print('training...'+'\n')
rf_model.fit(X_train,y_train)

with open('rf_model1.pkl', 'wb') as f:
    cPickle.dump(rf_model, f)
'''
with open('rf_model1.pkl', 'rb') as f:
    rf_model = cPickle.load(f)
'''
print('testing...'+'\n')
X_test,y_test=load_testingdata()
#print(rf_model.oob_score_)
correct=0
num0=0
num0total=0
num1=0
num1total=0
num2=0
num2total=0
Y=rf_model.predict(X_test)
for i in range(len(Y)):
	if(Y[i]==y_test[i]):
		correct+=1
		if(Y[i]==0):
			num0+=1
		if(Y[i]==1):
			num1+=1
		if(Y[i]==2):
			num2+=1
	if(y_test[i]==0):
		num0total+=1
	elif(y_test[i]==1):
		num1total+=1
	else:
		num2total+=1
print(float(num0)/float(num0total))
print(float(num1)/float(num1total))
print(float(num2)/float(num2total))
print(float(correct)/float(len(Y)))
