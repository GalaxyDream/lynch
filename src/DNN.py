from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.layers import Input, merge, Merge, LSTM, Bidirectional
from keras.utils import np_utils
from trainingData import load_trainingdata
from testingData import load_testingdata
from keras.models import model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint

nb_classes=3

a='loading data...'
print(a)

X_train,y_train=load_trainingdata()
X_test,y_test=load_testingdata()

print('X_train shape:')
print(X_train.shape)
print('y_train shape:')
print(len(y_train))
print('X_test shape:')
print(X_test.shape)
print('y_test shape:')
print(len(y_train))

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

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

'''
model.fit(X_train, Y_train, verbose=2, batch_size=128, nb_epoch=5000)
'''

'''checkpoint = ModelCheckpoint(filepath='{epoch:02d}-{val_loss:.2f}.hdf5')
model.fit(X=predictor_train, y=target_train, nb_epoch=5000,
         batch_size=128 ,validation_split=0.1, callbacks=[checkpoint])'''

checkpointer = ModelCheckpoint(filepath="./weights.hdf5",  monitor='val_acc',verbose=2, save_best_only=True)
model.fit(X_train, Y_train, batch_size=50, nb_epoch=5000, verbose=2,  validation_data=(X_test, Y_test), callbacks=[checkpointer])

score = model.evaluate(X_test, Y_test, batch_size=128, verbose=0)




print(score)

json_string = model.to_json()
open('new.json', 'w').write(json_string)
model.save_weights('new.h5')

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
Y=model.predict(X_test)
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
