# PROISE

This library implements the algorithms describes in the paper: "Probing machine-learning classifiers using noise, bubbles, and reverse correlation" by Etienne Thoret, Thomas Andrillon, Damien Leger, Daniel Pressnitzer (arxiv link)

It allows to probe the discriminative and canonical information of any trained machine learning classifier.

### Prerequisites
Depedencies: `tensorflow`, `numpy`, `matplotlib`, `keras`, `tensorflow`, `sklearn`, `random`

### Get started
Download or clone the repository

## Running the tests

The functions are all written in the `'./lib/proise.py'` file.

## Demo on MNIST
An use case on the database of handwritten digits MNIST is provided in `./demo.py` and summarized in below.

### Reverse correlation
The typical use of the reverse correlation method (Ahumada et al., 1976) uses the two following functions:
  1. Generate the probing samples: `proise.generateProbingSamples`
  2. Compute the canonical maps from the classifier predictions: `proise.ComputeCanonicalMaps`

For MNIST, it provides the following output:

![alt text](http://url/to/img.png)

(see `demo.py` for a detailed implementation)



### Bubbles
The typical use of the Bubbles method (Gosselin & Shyns, 2001) uses the three following functions:
  1. Generate bubble masks: `proise.generateBubbleMask`
  2. Generate the probing samples: `proise.generateProbingSamples`
  3. Compute the discriminative maps from the classifier predictions: `proise.ComputeDiscriminativeMaps`

For MNIST, it provides the following output:

![alt text](http://url/to/img.png)

(see `demo.py` for a detailed implementation)


## Authors

* **Etienne Thoret** - For any questions/suggestions/bugs: please contact me at <firstname><name>@gmail.com
