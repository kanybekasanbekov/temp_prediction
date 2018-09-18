from __future__ import print_function

import numpy as np
import tflearn
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')



GRID_ROW = 10
GRID_COL = 10

def rms_error(actual,prediction):
	mseData = []
	#error = 0
	for index in range(0,len(actual)):
		A = np.asarray(actual[index])
		B = np.asarray(predicted[index])
		error = (A-B)**2
		error = np.sum(error)
		error = np.sqrt(error)
		error = error/(GRID_ROW*GRID_COL)
		error = float(np.round(error, decimals=3))
		mseData.append(error)
		#print("\n",index+1,"'th error is:",error)

	return mseData




#importing training and testing data
#filenameTrainX = "filtered_images_train/ArrayOfTemps_X.npy"
#filenameTrainY = "filtered_images_train/ArrayOfTemps_Y.npy"

filenameTrainX = "filtered_ankara_train/ArrayOfTemps_X.npy"
filenameTrainY = "filtered_ankara_train/ArrayOfTemps_Y.npy"

filenameTestX = "filtered_ankara_test/ArrayOfTemps_X.npy"
filenameTestY = "filtered_ankara_test/ArrayOfTemps_Y.npy"


#dataTrainX => temp arrays for 6 days
dataTrainX = np.load(filenameTrainX)
#dataTrainY => temp array of the 7th day
dataTrainY = np.load(filenameTrainY)

#dataTestX => temp arrays for 6 days
dataTestX = np.load(filenameTestX)
#dataTestY => temp array of the 7th day
dataTestY = np.load(filenameTestY)

print("\n")
print("The number of train+validation samples are",len(dataTrainX))
print("The number of test samples are:", len(dataTestX))
print("\n")


#Building Neural Network
net = tflearn.input_data(shape=[None,6,100])#if 10*10, 6 days
#print(net.shape)
#print(tf.shape(net))
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 512, dropout=0.8)#n_units = 128
net = tflearn.fully_connected(net, 100, activation='relu')
#net = tf.cast(net, tf.int32)
net = tflearn.regression(net, optimizer='RMSprop', learning_rate=0.0005, loss='mean_square')



# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(dataTrainX, dataTrainY, validation_set=0.1, n_epoch=20, show_metric=True,
          batch_size=10)

# Count an Error
predicted = []

for i in range(len(dataTestX)):
	pred = model.predict([dataTestX[i]])[0]
	pred = [int(round(x)) for x in pred]
	predicted.append(pred)

actual = dataTestY.tolist()
"""
print("The number of actual samples are:", len(actual))
print("Data type of actual samples is:", type(actual))
print(actual)
print("\n*******************************************\n")
print("The number of predicted samples are:", len(predicted))
print("Data type of predicted samples is:", type(predicted))
print(predicted)
"""

#Counting an Error
error_list = rms_error(actual, predicted)
#print("This is ERROR list:\n",error_list)
error_array = np.asarray(error_list)
#plt.plot(error_array)
#plt.show()

model_error = np.mean(error_array)
model_error = float(np.round(model_error, decimals=2))
print("\nERROR of this model is:", model_error)

# Evaluate model

print("\n","******************************")

#score = model.evaluate(dataTestX, dataTestY,batch_size=21)
#print('Test accuracy: %0.4f%%' % (score[0] * 100))

# Run the model on one example
prediction = model.predict([dataTestX[0]])[0]
prediction = [int(round(x)) for x in prediction]
print("Prediction:\n",prediction)

print("\n\nActual output: \n",dataTestY[0])