# Comparative Analysis of ConvNet Architecture on Bird Species Dataset

This is an implementation of famous convolutional neural networks on Bird species dataset using Python 3, Keras, and Tensorflow. I trained these models and compared their results, and discussed how a model architecture can help the model success on specific data.


![birds](images/birds.png)

## Content of the repository

- [Getting Started](#getting-started)
- [Requirements](#requirements)
- [Bird Species Dataset](#bird-species-dataset)
- [Introduction to Convolutional Neural Networks Architecture](#introduction-to-convolutional-neural-networks-architecture)
    1. [AlextNet - 2012](#1-alextnet---2012)
    2. [VGG16 & VGG19 - 2014](#2-vgg16--vgg19---2014)
    3. [InceptionV3 - 2015](#3-inceptionv3---2015)
    4. [Residual Network or ResNet - 2016](#4-residual-network-or-resnet---2016)
    5. [MobileNet - 2017](#5-mobilenet---2017)
- [Results](#results)
- [Further Ideas](#further-ideas)
- [Refrences](#refrences)
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

DISCLAIMER : Some of the notes and defenitions in this section has taken from other articles, all of them are listed on [Refrences](#refrences)


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

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/alexnet.webp" style="display:block; margin-left: auto; margin-right: auto;" title="𝘈𝘳𝘤𝘩𝘪𝘵𝘦𝘤𝘵𝘶𝘳𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘈𝘭𝘦𝘹𝘕𝘦𝘵 𝘊𝘰𝘯𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯𝘢𝘭 𝘕𝘦𝘶𝘳𝘢𝘭 𝘕𝘦𝘵𝘸𝘰𝘳𝘬" />
    <span class="img_caption" style="display: block; text-align: center;">𝘈𝘳𝘤𝘩𝘪𝘵𝘦𝘤𝘵𝘶𝘳𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘈𝘭𝘦𝘹𝘕𝘦𝘵 𝘊𝘰𝘯𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯𝘢𝘭 𝘕𝘦𝘶𝘳𝘢𝘭 𝘕𝘦𝘵𝘸𝘰𝘳𝘬</span>
</span>


The model has five convolutional layers in the feature extraction part of the model and three fully connected layers in the classifier part of the model.

Input images were fixed to the size 224×224 with three color channels. In terms of the number of filters used in each convolutional layer, the pattern of increasing the number of filters with depth seen in LeNet was mostly adhered to, in this case, the sizes: 96, 256, 384, 384, and 256. Similarly, the pattern of decreasing the size of the filter (kernel) with depth was used, starting from the smaller size of 11×11 and decreasing to 5×5, and then to 3×3 in the deeper layers. Use of small filters such as 5×5 and 3×3 is now the norm.

A pattern of a convolutional layer followed by pooling layer was used at the start and end of the feature detection part of the model. Interestingly, a pattern of convolutional layer followed immediately by a second convolutional layer was used. This pattern too has become a modern standard.

### 2. VGG16 & VGG19 - 2014


Their architecture is generally referred to as VGG after the name of their lab, the Visual Geometry Group at Oxford. Their model was developed and demonstrated on the sameILSVRC competition, in this case, the ILSVRC-2014 version of the challenge.

The first important difference that has become a de facto standard is the use of a large number of small filters. Specifically, filters with the size 3×3 and 1×1 with the stride of one, different from the large sized filters in LeNet-5 and the smaller but still relatively large filters and large stride of four in AlexNet.

Max pooling layers are used after most, but not all, convolutional layers, learning from the example in AlexNet, yet all pooling is performed with the size 2×2 and the same stride, that too has become a de-facto standard. Specifically, the VGG networks use examples of two, three, and even four convolutional layers stacked together before a max-pooling layer is used. The rationale was that stacked convolutional layers with smaller filters approximate the effect of one convolutional layer with a larger sized filter, e.g., three stacked convolutional layers with 3×3 filters approximate one convolutional layer with a 7×7 filter.

Another important difference is the very large number of filters used. The number of filters increases with the depth of the model, although it starts at a relatively large number of 64 and increases through 128, 256, and 512 filters at the end of the feature extraction part of the model.

A number of variants of the architecture were developed and evaluated, although two are referred to most commonly given their performance and depth. They are named for the number of layers: they are the VGG-16 and the VGG-19 for 16 and 19 learned layers, respectively.

Below is a table is taken from the paper; note the two far right columns indicating the configuration (number of filters) used in the VGG-16 and VGG-19 versions of the architecture.

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/vgg.png" style="display:block; margin-left: auto; margin-right: auto;" title="𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑉𝐺𝐺 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘 (𝑡𝑎𝑘𝑒𝑛 𝑓𝑟𝑜𝑚 𝑡ℎ𝑒 2014 𝑝𝑎𝑝𝑒𝑟)" />
    <span class="img_caption" style="display: block; text-align: center;">𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑉𝐺𝐺 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘 (𝑡𝑎𝑘𝑒𝑛 𝑓𝑟𝑜𝑚 𝑡ℎ𝑒 2014 𝑝𝑎𝑝𝑒𝑟)</span>
</span>

The design decisions in the VGG models have become the starting point for the simple and direct use of convolutional neural networks in general.

Finally, the VGG work was among the first to release the valuable model weights under a permissive license that led to a trend among deep learning computer vision researchers. This, in turn, has led to the heavy use of pre-trained models like VGG in transfer learning as a starting point on new computer vision tasks.


### 3. InceptionV3 - 2015

The key innovation on the inception models is called the inception module. This is a block of parallel convolutional layers with different sized filters (e.g. 1×1, 3×3, 5×5) and a 3×3 max pooling layer, the results of which are then concatenated. Below is an example of the inception module taken from the paper.

A problem with a naive implementation of the inception model is that the number of filters (depth or channels) begins to build up fast, especially when inception modules are stacked.

Performing convolutions with larger filter sizes (e.g., 3 and 5) can be computationally expensive on a large number of filters. To address this, 1×1 convolutional layers are used to reduce the number of filters in the inception model. Specifically before the 3×3 and 5×5 convolutional layers and after the pooling layer. The image below taken from the paper shows this change to the inception module.


<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/inception_1.webp" style="display:block; margin-left: auto; margin-right: auto;" title="𝐸𝑥𝑎𝑚𝑝𝑙𝑒 𝑜𝑓 𝑡ℎ𝑒 𝐼𝑛𝑐𝑒𝑝𝑡𝑖𝑜𝑛 𝑀𝑜𝑑𝑢𝑙𝑒 𝑊𝑖𝑡ℎ 𝐷𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛𝑎𝑙𝑖𝑡𝑦 𝑅𝑒𝑑𝑢𝑐𝑡𝑖𝑜𝑛 (𝑡𝑎𝑘𝑒𝑛 𝑓𝑟𝑜𝑚 𝑡ℎ𝑒 2015 𝑝𝑎𝑝𝑒𝑟)" />
    <span class="img_caption" style="display: block; text-align: center;">𝐸𝑥𝑎𝑚𝑝𝑙𝑒 𝑜𝑓 𝑡ℎ𝑒 𝐼𝑛𝑐𝑒𝑝𝑡𝑖𝑜𝑛 𝑀𝑜𝑑𝑢𝑙𝑒 𝑊𝑖𝑡ℎ 𝐷𝑖𝑚𝑒𝑛𝑠𝑖𝑜𝑛𝑎𝑙𝑖𝑡𝑦 𝑅𝑒𝑑𝑢𝑐𝑡𝑖𝑜𝑛 (𝑡𝑎𝑘𝑒𝑛 𝑓𝑟𝑜𝑚 𝑡ℎ𝑒 2015 𝑝𝑎𝑝𝑒𝑟)</span>
</span>



A second important design decision in the inception model was connecting the output at different points in the model. This was achieved by creating small off-shoot output networks from the main network that were trained to make a prediction. The intent was to provide an additional error signal from the classification task at different points of the deep model in order to address the vanishing gradients problem. These small output networks were then removed after training.

Overall, Inception-v3 is a convolutional neural network architecture from the Inception family that makes several improvements, including using Label Smoothing, Factorized 7 x 7 convolutions, and the use of an auxiliary classifier to propagate label information lower down the network (along with the use of batch normalization for layers in the sidehead).

<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/inception_2.png" style="display:block; margin-left: auto; margin-right: auto;" title="𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝐼𝑛𝑐𝑒𝑝𝑡𝑖𝑜𝑛𝑉3 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘" />
    <span class="img_caption" style="display: block; text-align: center;">𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝐼𝑛𝑐𝑒𝑝𝑡𝑖𝑜𝑛𝑉3 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘</span>
</span>


### 4. Residual Network or ResNet - 2016

The residual networks model had an impressive 152 layers. Key to the model design is the idea of residual blocks that make use of shortcut connections. These are simply connections in the network architecture where the input is kept as-is (not weighted) and passed on to a deeper layer, e.g., skipping the next layer.

A residual block is a pattern of two convolutional layers with ReLU activation where the output of the block is combined with the input to the block, e.g. the shortcut connection. A projected version of the input used via 1×1 if the shape of the input to the block is different from the output of the block, so-called 1×1 convolutions. These are referred to as projected shortcut connections compared to the unweighted or identity shortcut connections.

The authors start with what they call a plain network, which is a VGG-inspired deep convolutional neural network with small filters (3×3), grouped convolutional layers followed with no pooling in between, and an average pooling at the end of the feature detector part of the model prior to the fully connected output layer with a softmax activation function.

The plain network is modified to become a residual network by adding shortcut connections in order to define residual blocks. Typically, the shape of the input for the shortcut connection is the same size as the residual block's output.

The image below was taken from the paper and from left to right compares the architecture of a VGG model, a plain convolutional model, and a version of the plain convolutional with residual modules called a residual network.


<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/resnet.webp" style="display:block; margin-left: auto; margin-right: auto;" title="𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑅𝑒𝑠𝑁𝑒𝑡 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘 𝑓𝑜𝑟 𝑖𝑚𝑎𝑔𝑒 𝑐𝑙𝑎𝑠𝑠𝑖𝑓𝑖𝑐𝑎𝑡𝑖𝑜𝑛" />
    <span class="img_caption" style="display: block; text-align: center;">𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑅𝑒𝑠𝑁𝑒𝑡 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘 𝑓𝑜𝑟 𝑖𝑚𝑎𝑔𝑒 𝑐𝑙𝑎𝑠𝑠𝑖𝑓𝑖𝑐𝑎𝑡𝑖𝑜𝑛</span>
</span>


We can summarize the key aspects of the architecture relevant in modern models as follows:

    - Use of shortcut connections.
    - Development and repetition of the residual blocks.
    - Development of very deep (152-layer) models.
    

### 5. MobileNet - 2017

MobileNets are built on depthwise separable convolution layers.Each depthwise separable convolution layer consists of a depthwise convolution and a pointwise convolution. Counting depthwise and pointwise convolutions as separate layers, a MobileNet has 28 layers. A standard MobileNet has 4.2 million parameters which can be further reduced by tuning the width multiplier hyperparameter appropriately.
The size of the input image is 224 × 224 × 3.

The detailed architecture of a MobileNet is given below :


<span class="img_container center" style="display: block;">
    <img alt="test" src="/images/mobilenet.png" style="display:block; margin-left: auto; margin-right: auto;" title="𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑀𝑜𝑏𝑖𝑙𝑒𝑁𝑒𝑡 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘" />
    <span class="img_caption" style="display: block; text-align: center;">𝐴𝑟𝑐ℎ𝑖𝑡𝑒𝑐𝑡𝑢𝑟𝑒 𝑜𝑓 𝑡ℎ𝑒 𝑀𝑜𝑏𝑖𝑙𝑒𝑁𝑒𝑡 𝐶𝑜𝑛𝑣𝑜𝑙𝑢𝑡𝑖𝑜𝑛𝑎𝑙 𝑁𝑒𝑢𝑟𝑎𝑙 𝑁𝑒𝑡𝑤𝑜𝑟𝑘</span>
</span>



MobileNets are a family of mobile-first computer vision models for TensorFlow, designed to effectively maximize accuracy while being mindful of the restricted resources for an on-device or embedded application.
MobileNets are small, low-latency, low-power models parameterized to meet the resource constraints of a variety of use cases. They can be built upon for classification, detection, embeddings, and segmentation.


## Results
 ### Training and Validation
 On the training data, as you can see, some models converge faster than others, like ResNet and MobileNet. However, faster convergence doesn't dedicate the highest accuracy (for example, VGG16 has more accuracy despite converging a bit later).
 
 On the bottom right plot, you can see the ResNet's validation loss doesn't come down as much as other models; this can indicate the model is overfitting on the training data.
 
 
 <span class="img_container center" style="display: block;">
    <img alt="test" src="/images/results.png" style="display:block; margin-left: auto; margin-right: auto;" title="𝘈𝘳𝘤𝘩𝘪𝘵𝘦𝘤𝘵𝘶𝘳𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘈𝘭𝘦𝘹𝘕𝘦𝘵 𝘊𝘰𝘯𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯𝘢𝘭 𝘕𝘦𝘶𝘳𝘢𝘭 𝘕𝘦𝘵𝘸𝘰𝘳𝘬" />
    <span class="img_caption" style="display: block; text-align: center;">𝘈𝘳𝘤𝘩𝘪𝘵𝘦𝘤𝘵𝘶𝘳𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘈𝘭𝘦𝘹𝘕𝘦𝘵 𝘊𝘰𝘯𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯𝘢𝘭 𝘕𝘦𝘶𝘳𝘢𝘭 𝘕𝘦𝘵𝘸𝘰𝘳𝘬</span>
</span>



 ### Test Accuracy and Loss
 
 <span class="img_container center" style="display: block;">
    <img alt="test" src="/images/test.jpg" style="display:block; margin-left: auto; margin-right: auto;" title="𝘈𝘳𝘤𝘩𝘪𝘵𝘦𝘤𝘵𝘶𝘳𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘈𝘭𝘦𝘹𝘕𝘦𝘵 𝘊𝘰𝘯𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯𝘢𝘭 𝘕𝘦𝘶𝘳𝘢𝘭 𝘕𝘦𝘵𝘸𝘰𝘳𝘬" />
    <span class="img_caption" style="display: block; text-align: center;">𝘈𝘳𝘤𝘩𝘪𝘵𝘦𝘤𝘵𝘶𝘳𝘦 𝘰𝘧 𝘵𝘩𝘦 𝘈𝘭𝘦𝘹𝘕𝘦𝘵 𝘊𝘰𝘯𝘷𝘰𝘭𝘶𝘵𝘪𝘰𝘯𝘢𝘭 𝘕𝘦𝘶𝘳𝘢𝘭 𝘕𝘦𝘵𝘸𝘰𝘳𝘬</span>
</span>



<p align="center">
<figure class="image">
  <img src="/images/test.jpg" alt="Architecture of the ResNet Convolutional Neural Network for image classification">
  <figcaption>Architecture of the ResNet Convolutional Neural Network for image classification</figcaption>
</figure>
</p>




## Further Ideas

The InceptionV3 model had a bad performance on the test set, and it seemed the model was overfitting the training set and couldn't perform well enough on the test set. My future goal is to figure out how to increase this model's accuracy and investigate the complicated structure of the Inception models and maybe tune the hyper-parameters better or find a more well-suited arrangement of layers. I will try to implement an inception model from scratch and get to know these models even better.


## Refrences

   ### Papers
   
> - [ImageNet Classification with Deep Convolutional Neural Networks, 2012.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
> - [Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014.](https://arxiv.org/abs/1409.1556)
> - [Going Deeper with Convolutions, 2015.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)
> - [Deep Residual Learning for Image Recognition, 2016.](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html)
> - [MobileNets: Efficient Convolutional Neural Networks, 2017.](https://arxiv.org/pdf/1704.04861.pdf)

  ### other refrences

> - [Wiki: Convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
> - [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)
> - [Convolutional Neural Network Model Innovations for Image Classification](https://machinelearningmastery.com/review-of-architectural-innovations-for-convolutional-neural-networks-for-image-classification/?unapproved=600474&moderation-hash=e231dae066102e3f68bd3a32ff68b640#comment-600474)
> - [Evolution of Convolutional Neural Network Architectures](https://medium.com/the-pen-point/evolution-of-convolutional-neural-network-architectures-6b90d067e403)