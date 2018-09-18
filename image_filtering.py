import cv2
import numpy as np 
import sys
import os, os.path
import glob

def is_black(pxl1, pxl2):
	if (pxl1[0] < pxl2[0]) & (pxl1[1] < pxl2[1]) & (pxl1[2] < pxl2[2]):
		return 1
	else:
		return 0

def is_white(pxl1, pxl2):
	if (pxl1[0] > pxl2[0]) & (pxl1[1] > pxl2[1]) & (pxl1[2] > pxl2[2]):
		return 1
	else:
		return 0		

def count_thresh(img):
	width, height = img.shape[:2]
	thresh = [0,0,0]

	for a in range(0,width):
		for b in range(0,height):
			thresh = thresh + img[a,b]
	num = width*height

	if num != 0:
		thresh = [thresh[0]/(width*height), thresh[1]/(width*height), thresh[2]/(width*height)]
	return thresh

def exist_black(img, thr):
	width, height = img.shape[:2]

	for x in range(0, width):
		for y in range(0, height):
			if is_black(img[x,y], thr):
				return 1
	return 0

def exist_white(img, thr):
	width, height = img.shape[:2]

	for x in range(0, width):
		for y in range(0, height):
			if is_white(img[x,y], thr):
				return 1
	return 0

def count_average(img, thr1, thr2):
	width, height = img.shape[:2]
	aver = [0,0,0]
	xyz = 0
	counter = 0

	for x1 in range(0, width):
		for y1 in range(0, height):
			if (is_black(img[x1,y1],thr1) | is_white(img[x1,y1], thr2)):
				xyz = 1
			else:
				counter = counter + 1
				aver = aver + img[x1,y1]

	if counter != 0:
		aver = [aver[0]/counter, aver[1]/counter, aver[2]/counter]
	return aver

def has_temp(img, thr1, thr2):
	width, height = img.shape[:2]
	count = 0

	for r in range(0, width):
		for t in range(0, height):
			if (is_black(img[r,t],thr1) | is_white(img[r,t], thr2)):
				count = count + 1

	qwer = (width*height)//2
	
	#print("Number of non temp pixels are:", count)
	#print("Number of all pixels is:", width*height)
	if count < qwer:
		return 1
	else:
		return 0

def make_ave(img, ave):
	width, height = img.shape[:2]

	for r1 in range(0, width):
		for t1 in range(0, height):
			img[r1,t1] = ave 

	return

def make_black(img):
	width, height = img.shape[:2]

	for r1 in range(0, width):
		for t1 in range(0, height):
			img[r1,t1] = [0,0,0]
	return

#return number of pixels which do not represent a temperature
def count_not_temp(img, thr1, thr2):
	width, height = img.shape[:2]
	num_pxl = width*height #total number of pixels
	count = 0 #counts number of "not temperature pixels"

	for r in range(0, width):
		for t in range(0, height):
			if (is_black(img[r,t],thr1) | is_white(img[r,t], thr2)):
				count = count + 1
	return count

def count_pxl(img, thr1, thr2):
	width, height = img.shape[:2]
	num_pxl = width*height #total number of pixels
	xyz = 0
	pxl_sum = [0,0,0]

	for r in range(0, width):
		for t in range(0, height):
			if (is_black(img[r,t],thr1) | is_white(img[r,t], thr2)):
				xyz = 1		
			else:
				pxl_sum = pxl_sum + img[r,t] 

	return pxl_sum

def fill_black_with_color(img, img1, img2, img3, img4, img5, img6, img7, img8, thr1, thr2, img_ave):
	width, height = img.shape[:2]
	pxl_num = width*height*9

	num = count_not_temp(img, thr1,thr2)
	num1 = count_not_temp(img1, thr1,thr2)
	num2 = count_not_temp(img2, thr1,thr2)
	num3 = count_not_temp(img3, thr1,thr2)
	num4 = count_not_temp(img4, thr1,thr2)
	num5 = count_not_temp(img5, thr1,thr2)
	num6 = count_not_temp(img6, thr1,thr2)
	num7 = count_not_temp(img7, thr1,thr2)
	num8 = count_not_temp(img8, thr1,thr2)

	total_num = num +num1 + num2 + num3 + num4 + num5 + num6 + num7 + num8

	if total_num <= pxl_num//2:
		num_pxl = count_pxl(img, thr1,thr2)
		num_pxl1 = count_pxl(img1, thr1,thr2)
		num_pxl2 = count_pxl(img2, thr1,thr2)
		num_pxl3 = count_pxl(img3, thr1,thr2)
		num_pxl4 = count_pxl(img4, thr1,thr2)
		num_pxl5 = count_pxl(img5, thr1,thr2)
		num_pxl6 = count_pxl(img6, thr1,thr2)
		num_pxl7 = count_pxl(img7, thr1,thr2)
		num_pxl8 = count_pxl(img8, thr1,thr2)

		total_num_pxl = num_pxl + num_pxl1 + num_pxl2 + num_pxl3 + num_pxl4 + num_pxl5 + num_pxl6 + num_pxl7 + num_pxl8
		temp_pxls = pxl_num - total_num
		color = [total_num_pxl[0]/temp_pxls,total_num_pxl[1]/temp_pxls,total_num_pxl[2]/temp_pxls]
		make_ave(img, color)
	else:
		make_ave(img, img_ave)

	return

def fill_black_with_color1(img, img1, img2, img3, thr1, thr2, img_ave):
	width, height = img.shape[:2]
	pxl_num = width*height*4

	num = count_not_temp(img, thr1,thr2)
	num1 = count_not_temp(img1, thr1,thr2)
	num2 = count_not_temp(img2, thr1,thr2)
	num3 = count_not_temp(img3, thr1,thr2)

	total_num = num +num1 + num2 + num3

	if total_num <= pxl_num//2:
		num_pxl = count_pxl(img, thr1,thr2)
		num_pxl1 = count_pxl(img1, thr1,thr2)
		num_pxl2 = count_pxl(img2, thr1,thr2)
		num_pxl3 = count_pxl(img3, thr1,thr2)

		total_num_pxl = num_pxl + num_pxl1 + num_pxl2 + num_pxl3
		temp_pxls = pxl_num - total_num
		color = [total_num_pxl[0]/temp_pxls,total_num_pxl[1]/temp_pxls,total_num_pxl[2]/temp_pxls]
		make_ave(img, color)
	else:
		make_ave(img, img_ave)

	return

def fill_black_with_color2(img, img1, img2, img3, img4, img5, thr1, thr2, img_ave):
	width, height = img.shape[:2]
	pxl_num = width*height*6

	num = count_not_temp(img, thr1,thr2)
	num1 = count_not_temp(img1, thr1,thr2)
	num2 = count_not_temp(img2, thr1,thr2)
	num3 = count_not_temp(img3, thr1,thr2)
	num4 = count_not_temp(img4, thr1,thr2)
	num5 = count_not_temp(img5, thr1,thr2)
	

	total_num = num +num1 + num2 + num3 + num4 + num5

	if total_num <= pxl_num//2:
		num_pxl = count_pxl(img, thr1,thr2)
		num_pxl1 = count_pxl(img1, thr1,thr2)
		num_pxl2 = count_pxl(img2, thr1,thr2)
		num_pxl3 = count_pxl(img3, thr1,thr2)
		num_pxl4 = count_pxl(img4, thr1,thr2)
		num_pxl5 = count_pxl(img5, thr1,thr2)

		total_num_pxl = num_pxl + num_pxl1 + num_pxl2 + num_pxl3 + num_pxl4 + num_pxl5
		temp_pxls = pxl_num - total_num
		color = [total_num_pxl[0]/temp_pxls,total_num_pxl[1]/temp_pxls,total_num_pxl[2]/temp_pxls]
		make_ave(img, color)
	else:
		make_ave(img, img_ave)

	return
		

def remove_black(img, thr):
	width, height = img.shape[:2]
	average = count_average(img, thr)
	#avr = (average + thr) / 2
	#thr = thr - [8,8,8]

	for a1 in range(0,width):
		for b1 in range(0,height):
			if is_black(img[a1,b1], thr):
				img[a1,b1] = average
	return


#*********************************************
def get_temp(img, pxl, listofcolor):
	#pxl = img[1,1]
	values = []

	if pxl[0] == 0.0 and pxl[1] == 0.0 and pxl[2] == 0.0:
		return "Does not have a temperature"

	for temp, color in listofcolor:
		woha = abs(pxl[0]-color[0])+abs(pxl[1]-color[1])+abs(pxl[2]-color[2])
		values.append(woha)
		
	minimum = min(values)
	order = values.index(minimum)
	return listofcolor[order][0]



list_color = []

n40_color = (-40, [204,204,204])
list_color.append(n40_color)
n39_color = (-39, [204,204,204])
list_color.append(n39_color)
n38_color = (-38, [204,204,204])
list_color.append(n38_color)
n37_color = (-37, [204,204,204])
list_color.append(n37_color)
n36_color = (-36, [204,204,204])
list_color.append(n36_color)
n35_color = (-35, [208,196,208])
list_color.append(n35_color)
n34_color = (-34, [213,187,214])
list_color.append(n34_color)
n33_color = (-33, [215,181,214])
list_color.append(n33_color)
n32_color = (-32, [213,175,216])
list_color.append(n32_color)
n31_color = (-31, [215,169,215])
list_color.append(n31_color)
n30_color = (-30, [213,163,214])
list_color.append(n30_color)
n29_color = (-29, [212,139,212])
list_color.append(n29_color)
n28_color = (-28, [214,112,214])
list_color.append(n28_color)
n27_color = (-27, [215,87,210])
list_color.append(n27_color)
n26_color = (-26, [215,62,214])
list_color.append(n26_color)
n25_color = (-25, [200,62,201])
list_color.append(n25_color)
n24_color = (-24, [187,63,187])
list_color.append(n24_color)
n23_color = (-23, [183,60,176])
list_color.append(n23_color)
n22_color = (-22, [168,63,166])
list_color.append(n22_color)
n21_color = (-21, [156,67,159])
list_color.append(n21_color)
n20_color = (-20, [148,66,148])
list_color.append(n20_color)
n19_color = (-19, [140,67,140])
list_color.append(n19_color)
n18_color = (-18, [135,65,135])
list_color.append(n18_color)
n17_color = (-17, [102,69,126])
list_color.append(n17_color)
n16_color = (-16, [77,68,121])
list_color.append(n16_color)
n15_color = (-15, [84,76,125])
list_color.append(n15_color)
n14_color = (-14, [88,80,127])
list_color.append(n14_color)
n13_color = (-13, [100,94,138])
list_color.append(n13_color)
n12_color = (-12, [111,104,146])
list_color.append(n12_color)
n11_color = (-11, [111,107,157])
list_color.append(n11_color)
n10_color = (-10, [113,109,168])
list_color.append(n10_color)
n9_color = (-9, [111,112,166])
list_color.append(n9_color)
n8_color = (-8, [109,114,170])
list_color.append(n8_color)
n7_color = (-7, [106,119,174])
list_color.append(n7_color)
n6_color = (-6, [101,124,178])
list_color.append(n6_color)
n5_color = (-5, [100,134,180])
list_color.append(n5_color)
n4_color = (-4, [95,143,181])
list_color.append(n4_color)
n3_color = (-3, [98,150,174])
list_color.append(n3_color)
n2_color = (-2, [100,156,169])
list_color.append(n2_color)
n1_color = (-1, [103,163,161])
list_color.append(n1_color)
zero_color = (0, [107,170,153])
list_color.append(zero_color)
p1_color = (1, [108,174,136])
list_color.append(p1_color)
p2_color = (2, [108,177,120])
list_color.append(p2_color)
p3_color = (3, [105,181,109])
list_color.append(p3_color)
p4_color = (4, [105,184,101])
list_color.append(p4_color)
p5_color = (5, [115,187,105])
list_color.append(p5_color)
p6_color = (6, [125,188,109])
list_color.append(p6_color)
p7_color = (7, [137,189,107])
list_color.append(p7_color)
p8_color = (8, [152,192,103])
list_color.append(p8_color)
p9_color = (9, [162,196,101])
list_color.append(p9_color)
p10_color = (10, [176,198,98])
list_color.append(p10_color)
p11_color = (11, [183,200,96])
list_color.append(p11_color)
p12_color = (12, [193,202,97])
list_color.append(p12_color)
p13_color = (13, [197,198,96])
list_color.append(p13_color)
p14_color = (14, [201,196,94])
list_color.append(p14_color)
p15_color = (15, [203,190,94])
list_color.append(p15_color)
p16_color = (16, [202,185,95])
list_color.append(p16_color)
p17_color = (17, [202,180,94])
list_color.append(p17_color)
p18_color = (18, [201,172,94])
list_color.append(p18_color)
p19_color = (19, [204,166,95])
list_color.append(p19_color)
p20_color = (20, [201,159,99])
list_color.append(p20_color)
p21_color = (21, [203,153,102])
list_color.append(p21_color)
p22_color = (22, [201,147,103])
list_color.append(p22_color)
p23_color = (23, [201,141,104])
list_color.append(p23_color)
p24_color = (24, [200,135,105])
list_color.append(p24_color)
p25_color = (25, [198,126,111])
list_color.append(p25_color)
p26_color = (26, [196,118,116])
list_color.append(p26_color)
p27_color = (27, [193,109,124])
list_color.append(p27_color)
p28_color = (28, [191,101,129])
list_color.append(p28_color)
p29_color = (29, [182,95,127])
list_color.append(p29_color)
p30_color = (30, [175,90,123])
list_color.append(p30_color)
p31_color = (31, [171,91,120])
list_color.append(p31_color)
p32_color = (32, [165,89,117])
list_color.append(p32_color)
p33_color = (33, [163,84,115])
list_color.append(p33_color)
p34_color = (34, [159,77,113])
list_color.append(p34_color)
p35_color = (35, [154,72,108])
list_color.append(p35_color)
p36_color = (36, [149,69,106])
list_color.append(p36_color)
p37_color = (37, [138,69,97])
list_color.append(p37_color)
p38_color = (38, [127,68,90])
list_color.append(p38_color)
p39_color = (39, [125,68,85])
list_color.append(p39_color)
p40_color = (40, [123,67,80])
list_color.append(p40_color)

#outfile = open("ArrayOfTemps.npy","w+")
outfileX = "ArrayOfTemps_X.npy" 
outfileY = "ArrayOfTemps_Y.npy"
outfileImage = "ArrayOfImages.npy"

#imageDir = "images/"
#imageDir = "2016-05-May/"
#imageDir = "2016-06-June/"
#imageDir = "2016JulyAugust/"
#imageDir = "Ankara_train/"
#imageDir = "Ankara_test/"
#imageDir = "2017-03-March/"
#imageDir = "Brazilia_test/"
#imageDir = "Brazilia_train/"
#imageDir = "Daejon_test/"
imageDir = "Daejon_train/"

#saveDir = "filtered_brazilia_train/"
#saveDir = "filtered_brazilia_test/"
#saveDir = "filtered_march/"
#saveDir = "filtered_ankara_val/"
#saveDir = "filtered_ankara_train/"
#saveDir = "filtered_images_train/"
#saveDir = "filtered_images_test/"
#saveDir = "filtered_images_test/"
#saveDir = "filtered_images/"
#saveDir = "filtered_daejon_test/"
saveDir = "filtered_daejon_train/"

image_list = []
suffix = ""
prefix = ""
image_format = ".jpg"
list_array = []
list_image = []
np_list_image = []

#for file in os.listdir(imageDir):
#	image_list.append(os.path.join(imageDir, file))

image_list = glob.glob(imageDir+"*.jpg")
image_list.sort()

for imagePath in image_list:
	#qwerty +=1
	array = []
	#img_title = input("Please input image title: ")
	#image_before_cropping = cv2.imread(img_title,-1)
	#outfile = "_ArrayOfTemps.npz" 
	#outfile = "_ArrayOfTemps.npy" 
	
	image_before_cropping = cv2.imread(imagePath)
	split_slash = imagePath.split("/")
	split_dot = split_slash[1].split(".")
	img_title = ""
	img_title = split_dot[0] 
	print(img_title+".jpg", "was opened.")

	if image_before_cropping is None:
		print("Error loading an image.")
	image = image_before_cropping[0:250, 0:350]

	#rangex = int(input("Input number for rangeX:(Note: it should divide 350) "))
	#rangey = int(input("Input number for rangeY:(Note: it should divide 250) "))
	rangex = 10
	rangey = 10

	#temp_array = np.ones((rangey, rangex),int)
	temp_array = []
	row = 0
	column = 0

	"""
	print("What do you want to do with 'bad' cells?:")
	print("Note: 'Bad' cell is the cell in which NOT temperature pixels are more than temperature pixels")

	black_color = input("Type 'A' if you want to fill them with approximate color, \nType 'B' if you want to fill them with black color:")
	"""
	black_color = "A"
	if (black_color == "A") | (black_color == "B"):
		abc = 1
	else:
		sys.exit("Ooops, you have entered wrong input. Enter 'A' or 'B'.")

	#cv2.imshow('Original Image', image_before_cropping)

	cell_num = rangex*rangey

	gridx = 350//rangex
	gridy = 250//rangey

	#gridx = gridx + 1
	#gridy = gridy + 1

	for j in range(0, rangey):
		for i in range(0, rangex):
			array.append(image[j*gridy:(j+1)*gridy, i*gridx:(i+1)*gridx])
			#array.append(image[i*gridx:(i+1)*gridx,j*gridy:(j+1)*gridy])

	list = []
	temp_list = []
	threshold_1 = [100, 100, 100]
	threshold_2 = [120, 120, 120]

	img_ave = count_average(image, threshold_1, threshold_2)

	#print("It's starting to print from top left corner.\n")
	for k in range(0, cell_num):
		thr_cell = count_thresh(array[k])
		th1 = [thr_cell[0]-10, thr_cell[1]-10, thr_cell[2]-10]
		th2 = [thr_cell[0]+10, thr_cell[1]+10, thr_cell[2]+10]

		if (threshold_1[0] > th1[0]) & (threshold_1[1] > th1[1]) & (threshold_1[2] > th1[2]):
			th1 = threshold_1
		if (threshold_2[0] < th2[0]) & (threshold_2[1] < th2[1]) & (threshold_2[2] < th2[2]):
			th2 = threshold_2

		if has_temp(array[k], th1, th2):
			average_cell = count_average(array[k], th1, th2)
			make_ave(array[k], average_cell)
		else:
			mod1 = k%rangex
			
			if black_color == "A":
				#prefix = "Filtered_Aproximate_"
				suffix = "_filtered_approximate"	

				if k == 0:
					fill_black_with_color1(array[k], array[k+1], array[k+rangex], array[k+rangex+1], threshold_1, threshold_2, img_ave)
				elif k == rangex-1:
					fill_black_with_color1(array[k], array[k-1], array[k+rangex-1], array[k+rangex], threshold_1, threshold_2, img_ave)
				elif (k > 0) & (k < rangex-1):
					fill_black_with_color2(array[k], array[k-1], array[k+1], array[k+rangex-1], array[k+rangex], array[k+rangex+1], threshold_1, threshold_2, img_ave)
				elif k == (rangey-1)*rangex:
					fill_black_with_color1(array[k], array[k-rangex], array[k-rangex+1], array[k+1], threshold_1, threshold_2, img_ave)
				elif k == rangey*rangex-1:
					fill_black_with_color1(array[k], array[k-rangex-1], array[k-rangex], array[k-1], threshold_1, threshold_2, img_ave)
				elif (k > (rangey-1)*rangex) & (k < rangey*rangex-1):
					fill_black_with_color2(array[k], array[k-rangex-1], array[k-rangex], array[k-rangex+1], array[k-1], array[k+1], threshold_1, threshold_2, img_ave)
				elif mod1 == 0:
					fill_black_with_color2(array[k], array[k-rangex], array[k-rangex+1], array[k+1], array[k+rangex], array[k+rangex+1], threshold_1, threshold_2, img_ave)
				elif mod1 == -1:
					fill_black_with_color2(array[k], array[k-rangex-1], array[k-rangex], array[k-1], array[k+rangex-1], array[k+rangex], threshold_1, threshold_2, img_ave)
				else:
					fill_black_with_color(array[k], array[k-rangex-1], array[k-rangex], array[k-rangex+1], array[k-1], array[k+1], array[k+rangex-1], array[k+rangex], array[k+rangex+1], threshold_1, threshold_2, img_ave)
			else:
				#prefix = "Filtered_Black_"
				suffix = "_filtered_black"

				make_black(array[k])

		row_column = divmod(k+1, rangex)
		clr = array[k][1,1]
		clr = [clr[2], clr[1], clr[0]]
		temper = get_temp(array[k], clr,list_color)
		
		if row_column[1] == 0:
			#print(k+1,"th cell (" ,row_column[0],"*",row_column[1]+rangex,") has temp value:", temper)
			row = row_column[0]-1
			column = row_column[1]+rangex-1
		else:
			#print(k+1,"th cell (" ,row_column[0]+1,"*",row_column[1],") has temp value:", temper)
			row = row_column[0]
			column = row_column[1]-1
		#print("It's color is: ", clr, "\n")
		
		if temper == "Does not have a temperature":
			temper = 999
		temp_array.append(temper)
		woha = (k, temper)
		temp_list.append(woha)

	#cv2.imshow('Filtered Image', image)

	#img_title = saveDir+img_title+suffix+".npy"
	#img_title = img_title+image_format
	list_array.append(temp_array)
	print(temp_array)
	np_list_image.append(image)
	
	#np.save(img_title, image)
	#cv2.imwrite(img_title, image)
	
	print("\n\n")
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

#np.savez(saveDir+outfile, *[list_array[x] for x in range(len(list_array))], )
#np.savez(outfile,temp_array)
#outfile.close() 


data_array_X = []
data_array_Y = []

for k in range(len(list_array)-6):
	dataX = []

	for l in range(k,k+6):
		dataX.append(list_array[l])

	dataY = list_array[k+6]

	data_array_X.append(dataX)
	data_array_Y.append(dataY)

image_data = []

for k in range(len(np_list_image)-6):
	dataX = []

	for l in range(k,k+6):
		dataX.append(np_list_image[l])

	image_data.append(dataX)

#np.save(filename, image_data)

#np.savez(saveDir+outfileX, *[data_array_X[x] for x in range(len(data_array_X))], )
#np.savez(saveDir+outfileY, *[data_array_Y[x] for x in range(len(data_array_Y))], )

np.save(saveDir+outfileX, data_array_X)
np.save(saveDir+outfileY, data_array_Y)
np.save(saveDir+outfileImage, image_data)

