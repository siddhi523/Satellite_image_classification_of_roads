# Import helpers and the ModelCNN class
from helpers import *
from cnn_model import cnn_model
from keras import models 
from keras import layers
# Instantiate the model
#model = cnn_model(shape = (150,150,3))
model = cnn_model(shape = (72,72,3))
# Load the model
model.load('final_model.h5')

# Print a summary to make sure the correct model is used
model.model.summary()

# We add all test images to an array, used later for generating a submission
image_filenames = []
for i in range(1, 51): #for i in range(1, 51):
    image_filename = 'provided/test_set_images/test_'+ str(i) +'/test_' + str(i) + '.png'
    image_filenames.append(image_filename)

# Set-up submission filename
submission_filename = 'final_submission.csv'

# Generates the submission
generate_submission(model, submission_filename, *image_filenames)

#test code###############################################################################
from keras.preprocessing import image 
import numpy as np 

# Pre-processing the image 
img = image.load_img("C:/Users/Admin/Desktop/New folder/Satellite-Image-Classification/provided/test_set_images/test_1/test_1.png", target_size = (72, 72)) 
img_tensor = image.img_to_array(img) 
img_tensor = np.expand_dims(img_tensor, axis = 0) 
img_tensor = img_tensor / 255.

# Print image tensor shape 
print(img_tensor.shape) 

# Print image 
import matplotlib.pyplot as plt 
plt.imshow(img_tensor[0]) 
plt.show() 


'''######################## part 2 ########################
# Outputs of the 8 layers, which include conv2D and max pooling layers 
layer_outputs = [layer.output for layer in model.model.layers[:24]] 
activation_model = models.Model(inputs = model.model.input, outputs = layer_outputs) 
activations = activation_model.predict(img_tensor) 

# Getting Activations of first layer 
first_layer_activation = activations[0] 

# shape of first layer activation 
print(first_layer_activation.shape) 

for i in range(1, 21):
	plt.matshow(activations[i][0, :, :, 6], cmap ='viridis') 
	plt.show()	
# 6th channel of the image after first layer of convolution is applied 
plt.matshow(first_layer_activation[0, :, :, 6], cmap ='viridis') 
plt.show()
# 15th channel of the image after first layer of convolution is applied 
plt.matshow(first_layer_activation[0, :, :, 15], cmap ='viridis') 
plt.show()'''