from __future__ import print_function

import numpy as np
import tflearn
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib


width = 350
height = 250
num_classes = 100

GRID_ROW = 10
GRID_COL = 10

def conv_net():
	network = tflearn.input_data(shape=[None,height,width,3])
	network = tflearn.conv_2d(network,96,11,strides=4,activation='relu')
	network = tflearn.max_pool_2d(network,3,strides=2)
	network = tflearn.conv_2d(network,96,11,strides=4,activation='relu')
	network = tflearn.max_pool_2d(network,3,strides=2)
	return network


def rms_error(actual,prediction):
	mseData = []
	for index in range(0,len(actual)):
		A = np.asarray(actual[index])
		B = np.asarray(predicted[index])
		error = (A-B)**2
		error = np.sum(error)
		error = np.sqrt(error)
		error = error/(GRID_ROW*GRID_COL)
		error = float(np.round(error, decimals=3))
		mseData.append(error)
		
	return mseData



#importing training and testing data
filenameTrainX = "filtered_ankara_train/newArrayOfImages.npy"
filenameTrainY = "filtered_ankara_train/ArrayOfTemps_Y.npy"

filenameTestX = "filtered_ankara_test/newArrayOfImages.npy"
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
print("The number of train+validation samples are",len(dataTrainX[0]))
print("The number of test samples are:", len(dataTestX[0]))
print("\n")



#Build a model
net1 = conv_net()
net2 = conv_net()
net3 = conv_net()
net4 = conv_net()
net5 = conv_net()
net6 = conv_net()

net = tflearn.merge([net1,net2,net3,net4,net5,net6],'concat')
net = tflearn.fully_connected(net,128,activation='relu')
net = tflearn.dropout(net,0.5)
net = tflearn.fully_connected(net,100,activation='relu')
net = tflearn.regression(net,optimizer='RMSprop',loss='mean_square',learning_rate=0.0005)


#Train a model
model = tflearn.DNN(net,tensorboard_verbose=0)
model.fit([dataTrainX[0],dataTrainX[1],dataTrainX[2],dataTrainX[3],dataTrainX[4],dataTrainX[5]],dataTrainY,
	validation_set=0.1, n_epoch=20, show_metric=True, batch_size=10)
print("**********After Fit *************")


# Count an Error
predicted = []

for i in range(len(dataTestX[0])):
	pred = model.predict([dataTestX[0],dataTestX[1],dataTestX[2],dataTestX[3],dataTestX[4],dataTestX[5]])[0]
	pred = [int(round(x)) for x in pred]
	predicted.append(pred)

actual = dataTestY.tolist()

#Counting an Error
error_list = rms_error(actual, predicted)
error_array = np.asarray(error_list)
model_error = np.mean(error_array)
model_error = float(np.round(model_error, decimals=2))
print("\nERROR of this model is:", model_error)


# Evaluate model
print("\n","******************************")

# Run the model on one example
prediction = model.predict([dataTestX[0],dataTestX[1],dataTestX[2],dataTestX[3],dataTestX[4],dataTestX[5]])[0]
prediction = [int(round(x)) for x in prediction]
print("Prediction:\n",prediction)
print("\n\nActual output: \n",dataTestY[0])