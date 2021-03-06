![Cleaning probabilistic data](Images/Process.png?raw=true "Cleaning probabilistic data")


## Project summary:
The code in this repository is used in the [Bachelor Thesis](https://github.com/RRMauritz/PDBQ/blob/master/Mauritz_BA_EEMCS.pdf) of Rutger Mauritz

## File descriptions:

`Runpannel`:
- From this file both the performance of the DAE and PIBN model are tested on several synthetic data sets
- The data-sets are loaded from the `\Datasets` directory

`DataQuality`:
- This file contains quality measures that are used for evaluation of the model performance
- Those quality measures are a.o. used in `Runpannel`

`DataPyAgrum`:
- Used for creating synthetic data sets via a Bayesian Network
- Contains an example of how a Bayesian Network can be constructed such that data can be sampled from it
- Used in `Runpannel`

`DataConversion`:
- Contains two methods:
	1) dataConversion: is used to add noise to the ground truth data and converts the synthetic data
	   to one-hot encoding like structures
	2) noise: is used to deliberately add noise to the already corrupted training data.
	   this is a part of the regularization of the DAE
- Used in AE_denoising

`BN_sup`:
- Implementation of the supervised PIBN model. It loads a pre-defined Bayesian Network (.bif file)
  and propagates the probabilistic data as virtual evidence through this BN, after which it extracts
  the posterior probabilities

`AE_denoising`:
- Implementation of the Denoising Autoencoder Model. This trained model is then applied to unseen
  corrupted test-data after which the resulting output is return, together with the training history

`Backend`:
- Contains a method jsd that is a Keras implementation of the Jensen Shannon Divergence
- Used in AE_denoising for training


## Directories

`\Archive`:
- Contains some extra files that are not necessary for the main process anymore. 

`\Datasets`: 
- Contains the data sets that are used in `runpannel`to evaluate the model performances.
- Each of these data sets contains  a description of its underling Bayesian Network (BN) model and comes with a .csv file containing the data, 
a .png file for a visualization of the BN network and a .bif file that can be used in PyAgrum. 

`\Results`: 
- A directory for storing the results of model evaluations. 

`\Images`: 
- Contains images that are used in the [Bachelor Thesis](https://github.com/RRMauritz/PDBQ/blob/master/Mauritz_BA_EEMCS.pdf)

## Needed packages:
- Numpy
- Pandas
- Keras
- PyAgrum: https://agrum.gitlab.io/pages/pyagrum.html
- Sklearn
------------------------------------------------------------------------

