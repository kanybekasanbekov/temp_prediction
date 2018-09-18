import importlib
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
#from resnet_sw import *

filename = "filtered_daejon_test/ArrayOfImages.npy"
savefile = "filtered_daejon_test/newArrayOfImages.npy"
data = np.load(filename)


"""
print("The number of train samples are:",len(data))
#print("The number images in one sample are:", len(data[0]))

image_data = []

for k in range(len(data)-6):
	dataX = []

	for l in range(k,k+6):
		dataX.append(data[l])

	image_data.append(dataX)

np.save(filename, image_data)
"""

new_data = [[],[],[],[],[],[]]
for i in range(len(data)):
	for j in range(len(data[0])):
		new_data[j].append(data[i][j])
np.save(savefile,new_data)





#print(data[0])
#cv2.imshow("Image",data[0])

#woha = data[0].flatten()
#print(woha)

#cv2.imshow("Image", woha)
#cv2.waitKey(0)
#cv2.destroyAllWindows()