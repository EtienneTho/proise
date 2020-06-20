# Basic functions for probing classifiers with revcor and bubbles
#
# Written by Etienne Thoret (2020)

import numpy as np
import random
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# compute the canonical map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneCanonicalMap(tab,y_pred,category):
	canonicalMap = (np.mean(np.asarray(tab[y_pred==category,:]),axis=0)-np.mean(np.asarray(tab[y_pred!=category,:]),axis=0))/np.sqrt(1/2*(np.std(np.asarray(tab[y_pred==category,:]),axis=0)**2+np.std(np.asarray(tab[y_pred!=category,:]),axis=0)**2))
	return canonicalMap

# compute the canonical maps for the different categories
def ComputeCanonicalMaps(tab,y_pred,dimOfinput):
	tabMaps = []
	for iCateg in range(np.unique(y_pred).shape[0]):
		category = np.unique(y_pred)[iCateg]
		tabMaps.append(ComputeOneCanonicalMap(tab,y_pred,category))
	return tabMaps

# compute the discriminative map for a given category (category) from the probed samples (tab) and their classification item (y_pred)
def ComputeOneDiscriminativeMap(tab,y_pred,category):
	dprimeTab = np.sum(np.asarray(tab[y_pred==category,:]),axis=0) / (np.sum(np.asarray(tab),axis=0))
	return dprimeTab

# compute the discriminative maps for the different categories
def ComputeDiscriminativeMaps(tab,y_pred,dimOfinput):
	tabMaps = []
	for iCateg in range(np.unique(y_pred).shape[0]):
		category = np.unique(y_pred)[iCateg]
		tabMaps.append(ComputeOneDiscriminativeMap(tab,y_pred,category))
	return tabMaps

# generate the bubble masks
def generateBubbleMask(dimOfinput=(28,28), nbMasks=10, nbBubbles=10, bubbleSize=[4, 4]):
	nFeatures = np.product(dimOfinput) # np.asarray(dimOfinput)[0]*np.asarray(dimOfinput)[1]
	masks = np.random.rand(nbMasks,nFeatures)

	for iSample in range(nbMasks):
	    vec = masks[iSample,:]
	    vec[:] = 0
	    bubblePos = (np.floor(np.random.rand(1,nbBubbles)*nFeatures)).astype(int)
	    vec[bubblePos] = 1
	    vec = np.reshape(vec,dimOfinput) 
	    vec = gaussian_filter(vec, sigma=bubbleSize)
	    masks[iSample,:] = vec.flatten()
	    masks[iSample,:] /= np.amax(masks[iSample,:])
	    masks[iSample,:] -= np.amin(masks[iSample,:])
	    masks[iSample,:] /= np.amax(masks[iSample,:])
	return masks

# generate probing samples from training set or with noise, or with pseudo-random noise from a testing set
def generateProbingSamples(x_train_set = [], x_test_set = [], dimOfinput=(28,28), bubbleMasks = [], probingMethod = 'bubbles', samplesMethod = 'gaussianNoise', nDim_pca = 50, nbRevcorTrials = 1000, normalizeNoiseCoeff = 10):
	if probingMethod == 'bubbles':
		N_probing_samples = bubbleMasks.shape[0]
		if samplesMethod == 'pseudoRandom':
			pca_noise = PCA(n_components= nDim_pca, svd_solver='auto', whiten=True).fit(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput))))
			probingSamples = pca_noise.inverse_transform(np.random.randn(bubbleMasks.shape[0],nDim_pca)) * bubbleMasks
		elif samplesMethod == 'gaussianNoise':
			probingSamples = np.random.randn(bubbleMasks.shape[0],bubbleMasks.shape[1]) * bubbleMasks
		elif samplesMethod == 'trainSet':
			probingSamples =  np.reshape(x_train_set,(x_train_set.shape[0],np.product(dimOfinput)))[0:bubbleMasks.shape[0],:] * bubbleMasks
	elif probingMethod == 'revcor':
		if samplesMethod == 'pseudoRandom':
			pca_noise = PCA(n_components= nDim_pca, svd_solver='auto', whiten=True).fit(np.reshape(x_test_set,(x_test_set.shape[0],np.product(dimOfinput))))
			probingSamples = pca_noise.inverse_transform(np.random.randn(nbRevcorTrials,nDim_pca))
		elif samplesMethod == 'gaussianNoise':
			probingSamples = np.random.randn(nbRevcorTrials,np.product(dimOfinput)) 
		elif samplesMethod == 'trainSet':
			probingSamples =  np.reshape(x_train_set,(x_train_set.shape[0],np.product(dimOfinput)))[0:nbRevcorTrials,:]  + np.random.randn(nbRevcorTrials,np.product(dimOfinput)) / normalizeNoiseCoeff
	return probingSamples





