# To Capture Frame
import cv2
import tensorflow as tf
# To process image array
import numpy as np


# import the tensorflow modules and load the model



# Attaching Cam indexed as 0, with the application software
vid = cv2.VideoCapture(0)
model = tf.keras.models.load_model("keras_model.h5")


# Infinite loop
while True:

	# Reading / Requesting a Frame from the Camera 
	status , frame = vid.read()

	# if we were sucessfully able to read the frame
	

		# Flip the frame
	frame = cv2.flip(frame , 1)
		
		
		
		#resize the frame
	img = cv2.resize(frame,(224,224))
	test_img = np.array(img,dtype=np.float32)
    
     	
		# expand the dimensions
	test_img = np.expand_dims(test_img,axis=0)
		# normalize it before feeding to the model
	normalize_img = test_img/255.0
		# get predictions from the model
	prediction = model.predict(normalize_img)
	print(prediction)
		
		
		# displaying the frames captured
	cv2.imshow('feed' , frame)

		# waiting for 1ms
	code = cv2.waitKey(1)
		
		# if space key is pressed, break the loop
	if code == 32:
			break

# release the camera from the application software
vid.release()

# close the open window
cv2.destroyAllWindows()
