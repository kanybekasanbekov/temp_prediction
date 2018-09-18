            Temperature prediction project
			    													
	Our task was to build several models which predict a temperature given
temperature of the previous 6 days. First one is LSTM model. It was trained
on sequential(non-image) data. Second and third models are CNN and CNN-LSTM. 
Both of them are trained on an image data.

	The most difficult part of this project was to collect data. 
Since we are using convolutional neural networks we needed image data,
historical temperature maps. I found in a web a website called VentuSky.com
which provides temperature maps from May of 2016 for every place in the Earth.
So, temperature maps were collected automatically from the website by using Xdotool script. 
Thus, I have collected data for 4 cities: Ankara(Turkey), Brazilia(Brazil), 
Daejeon(Korea) and Ulsan(Korea). 

	The next step was to filter those images. This was done with help 
of image filtering program. This program at first cuts bottom part of an 
image then divides the image into cells according to input(100 in our case)
and fills that cell with the average color of a cell. In the end, 
the program saves the filtered image in a numpy file, moreover, it also
 saves the temperature value of each cell in a different numpy file.

	After preparing the data, It was a time to build a model. 
I have used Tensorflow library called Tflearn to build models. It is very 
intuitive and easy to learn. So, I made 3 models: LSTM, CNN, and CNN-LSTM.
LSTM model gets input data and uses "lstm" function to find some dependencies 
in time series data and makes a prediction. On the other hand, CNN uses 
convolutional networks to find patterns in images and makes the prediction.
The third one, CNN-LSTM is a combination of previous models. 
It uses convolutional networks to find patterns and lstm to find 
dependencies between those patterns and makes the prediction. 
All the models were trained on data of 25 months(from May of 2016 to May of 2018)
and tested on a data of 3 months(from June to August of 2018).

	As a result LSTM model performed better, it has an error 0.16, 
whereas	CNN's error is 0.58 and CNN-LSTM's error is 0.88 respectively. 
Surprisingly the most complex model performed worse than others.
I think that the way how we built CNN-LSTM was a bit wrong and I can 
achieve better performance if I change its architecture. 
	



