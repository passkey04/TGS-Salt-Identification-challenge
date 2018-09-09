# TGS-Salt-Identification-challenge
#### Google colab, pytorch has been used
TGS Kaggle challenge, set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment.<br>In addition to the seismic images, the depth of the imaged location is provided for each image. The goal of the competition is to segment regions that contain salt.<br>
Dataset can be found [here](https://www.kaggle.com/c/tgs-salt-identification-challenge/data)
<br><br>
<b>In here U-Net algorithm is used to learn the mapping between seismic images and the salt filter mask</b><br>
U-net is a fully convolutional neural network used for image segmentation.<br>
-It's architecture uses the following types of layers:<br>
 - conv2D-simple convolutional layers,with padding and 3X3 kernel
 - MaxPooling2D-Simple maxpooling layers,2X2 kernel
 - Cropping2D-cropping layer used to crop feature maps and concatenate
 - Concatenate-layer used to concatenate multiplt feature maps from differnet stages of training
 - UpSampling2D-layer that increases size of feature map
 ## U-net is most appropriate for this problem because<br>
 We are doing multiclass classification, we are doing segmentation inside a single
image, U-net is proved to be standard architecture for computer vision when we
need not to segment the whole image by it's class but also to segment areas of 
image by class, i.e. it produces a mask.
It predicts a pixel wise segmentation of an input image, rather than the image
as a whole.This advantages over to predict different parts of the seismic image
i.e. salt or not-salt.
It designed like an autoencoder, the 1st block of layer is like an auto-encoder
This goes like U-shape, encoding path to all way up to decoding path<br>
## Here is the architecture of a U-net:<br>
![myimage-alt-tag](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png)
