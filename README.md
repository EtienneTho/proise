# PROISE

This library implements the algorithms describes in the paper: "Probing machine-learning classifiers using noise, bubbles, and reverse correlation" by Thoret, E., Andrillon, T., Leger, D., Pressnitzer, D. (arxiv link)

It allows to probe the discriminative and canonical information of any trained machine learning classifier.

### Prerequisites
Python3 depedencies: `tensorflow`, `numpy`, `matplotlib`, `keras`, `tensorflow`, `sklearn`, `random`

### Get started
  1. Download or clone the repository.
  2. All the functions are in the `'./lib/proise.py'` file and can be called by importing `lib` with `'from lib import proise'`

## Demo on MNIST
A use case on the database of handwritten digits MNIST is provided in `./demo.py` and summarized in below. The CNN trained on MNIST can be refitted with `'./fit_cnn_mnist.py'`

### Reverse correlation
The typical use of the reverse correlation method (Ahumada et al., 1976) uses the two following functions:
  1. Generate the probing samples: `proise.generateProbingSamples`
  2. Compute the canonical maps from the classifier predictions: `proise.ComputeCanonicalMaps`

For MNIST, it provides the following output:

![Canonical maps for each digit](https://github.com/EtienneTho/proise/blob/master/mnist_canonical.png)

(see `demo_mnist.py` for a detailed implementation)

### Bubbles
The typical use of the Bubbles method (Gosselin & Shyns, 2001) uses the three following functions:
  1. Generate bubble masks: `proise.generateBubbleMask`
  2. Generate the probing samples: `proise.generateProbingSamples`
  3. Compute the discriminative maps from the classifier predictions: `proise.ComputeDiscriminativeMaps`

For MNIST, it provides the following output:

![Discriminative maps for each digit](https://github.com/EtienneTho/proise/blob/master/mnist_discriminative.png)

(see `demo_mnist.py` for a detailed implementation)

## Reference
  * `Ahumada Jr, A., & Lovell, J. (1971). Stimulus features in signal detection. The Journal of the Acoustical Society of America, 49(6B), 1751-1756.`
  * `Gosselin, F., & Schyns, P. G. (2001). Bubbles: a technique to reveal the use of information in recognition tasks. Vision research, 41(17), 2261-2271.`

## Authors

* Developped by **Etienne Thoret** - For any questions/suggestions/bugs: please contact me at `firstnamename@gmail.com`
