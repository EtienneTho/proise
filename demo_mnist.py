# Basic functions for probing classifiers with reverse correlation and bubbles
# demo with mnist
#
# Written by Etienne Thoret (2020)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import model_from_json
from lib import proise


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
categories = range(10)
print(x_train.shape)
print(x_test.shape)

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
# # Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255.
x_test /= 255.
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])

# load the trained CNN
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model.h5")
print("Loaded model from disk")

#### BUBBLES
# generate bubble mask
nbMasks = 60000 # number of trials, 
nbBubbles = 10 # number of bubbles
dimOfinput = (28,28) # dimension of the input representation
bubbleSize = [4, 4] # bubbles standards deviations in the different dimensions
bubbleMasks = proise.generateBubbleMask(dimOfinput=dimOfinput, nbMasks=nbMasks, nbBubbles=nbBubbles, bubbleSize=bubbleSize)

# generate probing samples
probingMethod = 'bubbles' # choice of the probing method : bubbles or revcor
samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
nDim_pca = 50 # number of dimension to compute the PCA for the pseudo-random noise generation
probingSamples = proise.generateProbingSamples(x_train_set = x_train, x_test_set = x_test, dimOfinput=dimOfinput, bubbleMasks = bubbleMasks, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca)
probingSamples = np.reshape(probingSamples,(probingSamples.shape[0],28,28,1))

# compute predictions with the trained classifier
y_pred = model.predict_classes(probingSamples)

# compute discriminative maps
discriminativeMaps = proise.ComputeDiscriminativeMaps(bubbleMasks,y_pred, dimOfinput=dimOfinput)

# plot diagnostic maps
fig, ax = plt.subplots(nrows=4, ncols=3)
for iDigit in range(np.asarray(discriminativeMaps).shape[0]):
	toPlot = np.asarray(discriminativeMaps)[iDigit,:]
	plt.subplot(4,3,iDigit+1)
	im = plt.imshow(np.reshape((toPlot),(28,28)) ,cmap='coolwarm') 
	plt.title(str(np.unique(y_pred)[iDigit]))
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')	
	plt.tight_layout()
for x in ax.ravel():
    x.axis("off")
plt.savefig('mnist_discriminative.png')
plt.show()

#### REVERSE CORRELATION
# generate probing samples
dimOfinput = (28,28) # dimension of the input representation
probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
nbRevcorTrials = 60000 # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
nDim_pca = 50 # number of dimension to compute the PCA for the pseudo-random noise generation
probingSamples = proise.generateProbingSamples(x_train_set = x_train, x_test_set = x_test, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
probingSamples = np.reshape(probingSamples,(probingSamples.shape[0],28,28,1))

# compute predictions with the trained classifier
y_pred = model.predict_classes(probingSamples)

canonicalMaps = proise.ComputeCanonicalMaps(probingSamples,y_pred, dimOfinput=dimOfinput)

# plot diagnostic maps
fig, ax = plt.subplots(nrows=4, ncols=3)
for iDigit in range(np.asarray(canonicalMaps).shape[0]):
	toPlot = np.asarray(canonicalMaps)[iDigit,:]
	plt.subplot(4,3,iDigit+1)
	im = plt.imshow(np.reshape((toPlot),(28,28)) ,cmap='coolwarm') 
	plt.title(str(np.unique(y_pred)[iDigit]))
	plt.xticks([])
	plt.yticks([])
	plt.axis('off')	
	plt.tight_layout()
for x in ax.ravel():
    x.axis("off")
plt.savefig('mnist_canonical.png')
plt.show()


