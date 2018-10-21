# Image classifier project

## 1. Installation
I use python 3.5 to create this project and the libraries I used are:
 - Pandas
 - Numpy
 - Matplotlib
 - Scikit-Learn
 - PyTorch
 - torchvision
 - PIL
 - json
 - collections


## 2. Project Motivation

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I first develop code for an image classifier built with PyTorch, then convert it into a command line application. I train an image classifier to recognize different species of flowers. It is like a phone app that tells you the name of the flower the camera is looking at. In practice I train this classifier, then export it for use in the application.


## 3. File Descriptions
  - Image Classifier Project.ipynb
      - This is image classifier to recognize different species of flowers.
  - cat_to_name.json	 
  - predict.py	 
  - readme.md	 
  - train.py
  - utils.py

## 4. Summary

I use one of the pretrained models from torchvision.models(like 'densenet121' 'vgg16' vgg19_bn') to get the image features. Build and train a new feed-forward classifier using those features.

I use the learning transfer technology to do the following step:
  - Load a pre-trained network (the VGG networks work great and are straightforward to use)
  - Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
  - Train the classifier layers using backpropagation using the pre-trained network to get the features
  - Track the loss and accuracy on the validation set to determine the best hyperparameters

When training with GPU I only update the weights of the feed-forward network. I  get the validation accuracy above 82% result. I try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next.

## 5. Licensing, Author, Acknowledgements
    This work is licensed under a [Creative Commons  Attribution-NonCommercial-NoDerivatives 4.0 International License](http://creativecommons.org/licenses/by-nc-nd/4.0/). Please refer to [Udacity Terms of Service](https://www.udacity.com/legal) for further information.
