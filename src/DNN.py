from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Input, merge, Merge, LSTM, Bidirectional
from keras.utils import np_utils
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd

nb_classes=3

a='loading data...'
print(a)


#import data to "raw_data"
raw_data=pd.read_csv('../data/lynch.csv')


#load training data and label
X_train=np.loadtxt('../data/training.out',delimiter=',')
y_train=raw_data['xinsong'].iloc[0:800]

#load testing data and label
X_test=np.loadtxt('../data/testing.out',delimiter=',')
y_test=raw_data['xinsong'].iloc[800:1000]



#X_train,y_train=load_trainingdata()
#X_test,y_test=load_testingdata()

print('X_train shape:')
print(X_train.shape)
print('y_train shape:')
print(len(y_train))
print('X_test shape:')
print(X_test.shape)
print('y_test shape:')
print(len(y_train))

#transfer labels to one-hot vectors
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#build model
b='training...'
print(b)
model = Sequential()
model.add(Dense(600, input_shape=(300,)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2400))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(1200))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(600))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(300))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


#training the model with checkpointer
checkpointer = ModelCheckpoint(filepath="./weights.hdf5",  monitor='val_acc',verbose=2, save_best_only=True)
model.fit(X_train, Y_train, batch_size=50, nb_epoch=30, verbose=2,  validation_data=(X_test, Y_test), callbacks=[checkpointer])

score = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)

print(score)

#save model
json_string = model.to_json()
open('new.json', 'w').write(json_string)
model.save_weights('new.h5')

#test the model
if __name__ == '__main__':
	print('testing...'+'\n')
	correct=0
	num0=0          #the number of label 0 which is correctly identified
	num0total=0	#the total number of label 0
	num1=0		#the number of label 1 which is correctly identified
	num1total=0	#the total number of label 1
	num2=0		#the number of label 2 which is correctly identified
	num2total=0	#the total number of label2
	Y_prob=model.predict(X_test)
	Y_predict=[]
	for i in range(len(Y_prob)):
		if (Y_prob[i][0]>Y_prob[i][1] and Y_prob[i][0]>Y_prob[i][2]): 
			Y_predict.append(0)
		elif (Y_prob[i][1]>Y_prob[i][0] and Y_prob[i][1]>Y_prob[i][2]): Y.append(1)
		else: Y_predict.append(2)

	for i in range(len(Y_predict)):
		if(Y_predict[i]==y_test.iloc[i]):
			correct+=1
			if(Y_predict[i]==0):
				num0+=1
			if(Y_predict[i]==1):
				num1+=1
			if(Y_predict[i]==2):
				num2+=1
		if(y_test.iloc[i]==0):
			num0total+=1
		elif(y_test.iloc[i]==1):
			num1total+=1
		else:
			num2total+=1
	print(float(num0)/float(num0total)) #the accuracy for label 0
	print(float(num1)/float(num1total)) #the accuracy for label 1
	print(float(num2)/float(num2total)) #the accuracy for label 2
	print(float(correct)/float(len(Y))) #the accuracy for testset

