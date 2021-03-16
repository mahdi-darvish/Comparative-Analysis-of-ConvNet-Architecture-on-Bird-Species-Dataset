# Comparative Analysis of ConvNet Architecture on Bird Species Dataset

This is an implementation of famous convolutional neural networks on Bird species dataset using Python 3, Keras, and Tensorflow. I trained these models and compared their results, and discussed how a model architecture can help the model success on specific data.


![birds](images/birds.png)

## Content of the repository

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Bird Species Dataset](#bird-species-dataset)
- [Introduction to Convolutional Neural Networks Architecture](#introduction-to-convolutional-neural-networks-architecture)
    1. [AlextNet - 2012](#1-alextnet---2012)
    2. [VGG16 & VGG19 - 2014]()
    3. [InceptionV3 - 2015]()
    4. [Residual Network or ResNet - 2016]()
    5. [MobileNet - 2017]()
- [Results](#results)
## Getting Started

- [Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset.ipynb](https://github.com/mahdi-darvish/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset/blob/main/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset.ipynb) This is the main notebook containing python code to train the models and visualize the results.
- [requirements.txt](https://github.com/mahdi-darvish/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset/blob/main/requirements.txt) shows the required packages and dependencies to run this project.
- [/images](https://github.com/mahdi-darvish/Comparative-Analysis-of-ConvNet-Architecture-on-Bird-Species-Dataset/tree/main/images) Containing screenshots used throughout the repo.

- *note*: dataset files are not uploaded here. You may check this section to find a guide on how to access them.

## Requirements

Python 3.8.6, TensorFlow 2.4.1, Keras 2.4.3, and other common packages listed in `requirements.txt`.

You can use `$ pip install -r requirements.txt` inside your virtual environment to install them all or do it manually.


## Bird Species Dataset

Data set of 250 bird species. 35215 training images, 1250 test images(5 per species) and 1250 validation images(5 per species.
All images are 224 X 224 X 3 color images in jpg format. Also includes a "consolidated" image set that combines the training, test and validation images into a single data set. This is useful for users that want to create their own training, test and validation sets. Each set contains 250 sub directories, one for each bird species.

The dataset is available on Kaggle. You can find more information and download the data [here](https://www.kaggle.com/gpiosenka/100-bird-species).

## Introduction to Convolutional Neural Networks Architecture

DISCLAIMER : some of the notes and defenitions in this section has taken from other articles, all of them are listed on [Refrences](#refrences)


Convolutional neural networks are comprised of two specific elements, namely convolutional layers and pooling layers.

Although simple, there are near-infinite ways to arrange these layers for a given computer vision problem.

Fortunately, there are both common patterns for configuring these layers and architectural innovations that you can use to develop very deep convolutional neural networks. Studying these architectural design decisions developed for state-of-the-art image classification tasks can provide both a rationale and intuition for using these designs when designing your own deep convolutional neural network models.

The elements of a convolutional neural network, such as convolutional and pooling layers, are relatively straightforward to understand.

The challenging part of using convolutional neural networks in practice is how to design model architectures that best use these simple elements.

A useful approach to learning how to design effective convolutional neural network architectures is to study successful applications. This is particularly straightforward to do because of the intense study and application of CNNs through 2012 to 2017 for the ImageNet Large Scale Visual Recognition Challenge, or ILSVRC. This challenge resulted in both the rapid advancement in the state of the art for very difficult computer vision tasks and the development of general innovations in the architecture of convolutional neural network models.

By understanding these milestone models and their architecture or architectural innovations from a high-level, you will develop both an appreciation for the use of these architectural elements in modern applications of CNN in computer vision, and be able to identify and choose architecture elements that may be useful in the design of your own models.

Let's dive deeper into how each of these models works and break down their architecture to get a better intuition.


### 1. AlextNet - 2012

AlexNet made use of the rectified linear activation function, or ReLU, as the nonlinearly after each convolutional layer, instead of S-shaped functions such as the logistic or tanh that were common up until that point. Also, a softmax activation function was used in the output layer, now a staple for multi-class classification with neural networks.

The average pooling used in LeNet-5 was replaced with a max pooling method, although in this case, overlapping pooling was found to outperform non-overlapping pooling that is commonly used today (e.g. stride of pooling operation is the same size as the pooling operation, e.g. 2 by 2 pixels). To address overfitting, the newly proposed dropout method was used between the fully connected layers of the classifier part of the model to improve generalization error.

<figure class="image">
  <img src="/images/alexnet.webp" alt="Architecture of the AlexNet Convolutional Neural Network">
  <figcaption>Architecture of the AlexNet Convolutional Neural Network</figcaption>
</figure>

The model has five convolutional layers in the feature extraction part of the model and three fully connected layers in the classifier part of the model.

Input images were fixed to the size 224×224 with three color channels. In terms of the number of filters used in each convolutional layer, the pattern of increasing the number of filters with depth seen in LeNet was mostly adhered to, in this case, the sizes: 96, 256, 384, 384, and 256. Similarly, the pattern of decreasing the size of the filter (kernel) with depth was used, starting from the smaller size of 11×11 and decreasing to 5×5, and then to 3×3 in the deeper layers. Use of small filters such as 5×5 and 3×3 is now the norm.

A pattern of a convolutional layer followed by pooling layer was used at the start and end of the feature detection part of the model. Interestingly, a pattern of convolutional layer followed immediately by a second convolutional layer was used. This pattern too has become a modern standard.

### 2. VGG16 & VGG19 - 2014


Their architecture is generally referred to as VGG after the name of their lab, the Visual Geometry Group at Oxford. Their model was developed and demonstrated on the sameILSVRC competition, in this case, the ILSVRC-2014 version of the challenge.

The first important difference that has become a de facto standard is the use of a large number of small filters. Specifically, filters with the size 3×3 and 1×1 with the stride of one, different from the large sized filters in LeNet-5 and the smaller but still relatively large filters and large stride of four in AlexNet.

Max pooling layers are used after most, but not all, convolutional layers, learning from the example in AlexNet, yet all pooling is performed with the size 2×2 and the same stride, that too has become a de facto standard. Specifically, the VGG networks use examples of two, three, and even four convolutional layers stacked together before a max pooling layer is used. The rationale was that stacked convolutional layers with smaller filters approximate the effect of one convolutional layer with a larger sized filter, e.g. three stacked convolutional layers with 3×3 filters approximates one convolutional layer with a 7×7 filter.

Another important difference is the very large number of filters used. The number of filters increases with the depth of the model, although starts at a relatively large number of 64 and increases through 128, 256, and 512 filters at the end of the feature extraction part of the model.

A number of variants of the architecture were developed and evaluated, although two are referred to most commonly given their performance and depth. They are named for the number of layers: they are the VGG-16 and the VGG-19 for 16 and 19 learned layers respectively.

Below is a table taken from the paper; note the two far right columns indicating the configuration (number of filters) used in the VGG-16 and VGG-19 versions of the architecture.

<figure class="image">
  <img src="/images/vgg.png" alt="Architecture of the VGG Convolutional Neural Network (taken from the 2014 paper).">
  <figcaption>Architecture of the AlexNet Convolutional Neural Network</figcaption>
</figure>

The design decisions in the VGG models have become the starting point for simple and direct use of convolutional neural networks in general.

Finally, the VGG work was among the first to release the valuable model weights under a permissive license that led to a trend among deep learning computer vision researchers. This, in turn, has led to the heavy use of pre-trained models like VGG in transfer learning as a starting point on new computer vision tasks.


### 3. InceptionV3 - 2015

The key innovation on the inception models is called the inception module. This is a block of parallel convolutional layers with different sized filters (e.g. 1×1, 3×3, 5×5) and a 3×3 max pooling layer, the results of which are then concatenated. Below is an example of the inception module taken from the paper.

A problem with a naive implementation of the inception model is that the number of filters (depth or channels) begins to build up fast, especially when inception modules are stacked.

Performing convolutions with larger filter sizes (e.g. 3 and 5) can be computationally expensive on a large number of filters. To address this, 1×1 convolutional layers are used to reduce the number of filters in the inception model. Specifically before the 3×3 and 5×5 convolutional layers and after the pooling layer. The image below taken from the paper shows this change to the inception module.


<figure class="image">
  <img src="/images/inception_1.png" alt="Example of the Inception Module With Dimensionality Reduction (taken from the 2015 paper).">
  <figcaption>Example of the Inception Module With Dimensionality Reduction (taken from the 2015 paper).</figcaption>
</figure>



A second important design decision in the inception model was connecting the output at different points in the model. This was achieved by creating small off-shoot output networks from the main network that were trained to make a prediction. The intent was to provide an additional error signal from the classification task at different points of the deep model in order to address the vanishing gradients problem. These small output networks were then removed after training.

Overall, Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifer to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).


<figure class="image">
  <img src="/images/inception_2.png" alt="Architecture of the InceptionV3 Convolutional Neural Network">
  <figcaption>Architecture of the InceptionV3 Convolutional Neural Network</figcaption>
</figure>


### 4. Residual Network or ResNet - 2016

Residual networks model had an impressive 152 layers. Key to the model design is the idea of residual blocks that make use of shortcut connections. These are simply connections in the network architecture where the input is kept as-is (not weighted) and passed on to a deeper layer, e.g. skipping the next layer.

A residual block is a pattern of two convolutional layers with ReLU activation where the output of the block is combined with the input to the block, e.g. the shortcut connection. A projected version of the input used via 1×1 if the shape of the input to the block is different to the output of the block, so-called 1×1 convolutions. These are referred to as projected shortcut connections, compared to the unweighted or identity shortcut connections.

The authors start with what they call a plain network, which is a VGG-inspired deep convolutional neural network with small filters (3×3), grouped convolutional layers followed with no pooling in between, and an average pooling at the end of the feature detector part of the model prior to the fully connected output layer with a softmax activation function.

The plain network is modified to become a residual network by adding shortcut connections in order to define residual blocks. Typically the shape of the input for the shortcut connection is the same size as the output of the residual block.

The image below was taken from the paper and from left to right compares the architecture of a VGG model, a plain convolutional model, and a version of the plain convolutional with residual modules, called a residual network.


<figure class="image">
  <img src="/images/resnet.webp" alt="Architecture of the ResNet Convolutional Neural Network for image classification">
  <figcaption>Architecture of the ResNet Convolutional Neural Network for image classification</figcaption>
</figure>

We can summarize the key aspects of the architecture relevant in modern models as follows:

    -Use of shortcut connections.
    -Development and repetition of the residual blocks.
    -Development of very deep (152-layer) models.
    

### 5. MobileNet - 2017

MobileNets are built on depthwise seperable convolution layers.Each depthwise seperable convolution layer consists of a depthwise convolution and a pointwise convolution.Counting depthwise and pointwise convolutions as seperate layers, a MobileNet has 28 layers.A standard MobileNet has 4.2 million parameters which can be further reduced by tuning the width multiplier hyperparameter appropriately.
The size of the input image is 224 × 224 × 3.

The detailed architecture of a MobileNet is given below :

<figure class="image">
  <img src="/images/mobilenet.png" alt="Architecture of the MobileNet Convolutional Neural Network">
  <figcaption>Architecture of the MobileNet Convolutional Neural Network</figcaption>
</figure>

MobileNets are a family of mobile-first computer vision models for TensorFlow, designed to effectively maximize accuracy while being mindful of the restricted resources for an on-device or embedded application.
MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings, and segmentation.


## Results





## Conclution


## Refrences

## Further Ideas