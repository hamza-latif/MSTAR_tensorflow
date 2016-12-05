MSTAR Tensorflow

Synthetic Aperture Radar (SAR) object recognition is an important problem for automatic target recognition and aerial reconnaissance in military applications. We propose to use a deep convolutional neural network to classify and extract useful features of target chips taken from SAR image scenes. We will use the publically available Moving and Stationary Target Acquisition and Recognition (MSTAR) database as our dataset to evaluate our network.

![MSTAR image](mstar_image.png "Sample MSTAR image")

# Introduction

We want to train a deep neural network to identify targets in the three class MSTAR dataset obtained from https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=targets and possibly the ten class dataset from 
https://www.sdms.afrl.af.mil/index.php?collection=mstar&page=mixed.

Our base will be the paper *Deep convolutional neural networks for ATR from SAR imagery* [Morgan 2015], where they claim to achieve an overall 92.3% classification accuracy for the ten class problem.

We will explore a few number of different convolutional network configurations and also residual networks.

[//]: # (Also look at *APPLICATION OF DEEP LEARNING ALGORITHMS TO MSTAR DATA* [Wang, Chen, Xu, Jin 2015] where they claim 99.1% accuracy with All-ConvNets)

# Background

## Convolutional Networks

There are lots of tutorials on convolutional networks. Google them.

## Residual Networks

Residual networks are a recent evolution of convolutional networks that have allowed much deeper networks than conventional convolutional networks. For example, in the paper *Deep Residual Learning for Image Recognition* [He, Zhang, Ren, Sun 2015] they used an ensemble of 6 residual networks, each having up to 152 layers, to achieve a 3.57% top-5 error rate in ILSVRC 2015.
