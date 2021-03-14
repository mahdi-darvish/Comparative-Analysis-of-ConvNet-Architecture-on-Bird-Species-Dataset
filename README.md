# Comparative Analysis of ConvNet Architecture on Bird Species Dataset

This is an implementation of famous convolutional neural networks on Bird species dataset using Python 3, Keras, and Tensorflow. I trained these models and compared their results, and discussed how a model architecture can help the model success on specific data.


![birds](images/birds.png)

## Content of the repository

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Bird Species Dataset](#bird-species-dataset)
- [Introduction to Convolutional Neural Networks Architecture](#introduction-to-convolutional-neural-networks-architecture)
    - [1. AlextNet - 2012]()


## Getting Started

- [Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset.ipynb](https://github.com/mahdi-darvish/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset/blob/main/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset.ipynb) This is the main notebook containing python code to train the models and visualize the results.
- [requirements.txt](https://github.com/mahdi-darvish/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset/blob/main/requirements.txt) shows the required packages and dependencies to run this project.
- [/images](https://github.com/mahdi-darvish/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset/tree/main/images) Containing screenshots used throughout the repo.

- *note*: dataset files are not uploaded here. You may check this section to find a guide on how to access them.

## Requirements

Python 3.8.6, TensorFlow 2.4.1, Keras 2.4.3, and other common packages listed in `requirements.txt`.

You can use `$ pip install -r requirements.txt` inside your virtual environment to install them all or do it manually.


## Bird Species Dataset

## Introduction to Convolutional Neural Networks Architecture

DISCLAIMER : some of the notes and defenitions in this section has taken from other articles, all of them are listed on [Refrences](#refrences)


Convolutional neural networks are comprised of two specific elements, namely convolutional layers and pooling layers.

Although simple, there are near-infinite ways to arrange these layers for a given computer vision problem.

Fortunately, there are both common patterns for configuring these layers and architectural innovations that you can use to develop very deep convolutional neural networks. Studying these architectural design decisions developed for state-of-the-art image classification tasks can provide both a rationale and intuition for using these designs when designing your own deep convolutional neural network models.

Let's dive deeper into how each of these models works and break down their architecture to get a better intuition.


## 1. AlextNet - 2012



