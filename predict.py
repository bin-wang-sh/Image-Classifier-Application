import matplotlib.pyplot as plt
import json
import numpy as np
import torch
from torch import optim,nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,models,transforms
import argparse
from PIL import Image
from utils import label_map,load_pretrained_model,set_device
def predict_args():
    #Basic usage: python predict.py /path/to/image checkpoint
    parser = argparse.ArgumentParser( description='Predict Images' )
    parser.add_argument('image_path', type=str,
                     help = "Data source directory")
    parser.add_argument('checkpoint', default='vgg16.pth', type=str,
                     help = "Select pretrained  model")
    parser.add_argument('--category_names', default="cat_to_name.json",type=str,
                     help="Use a mapping of categories to real names")
    parser.add_argument('--top_k', default=5, type=int,
                     help="Return top KK")
    parser.add_argument('--gpu', default=False, action='store_true',
                     help="use --gpu to enable GPU process")
    return parser.parse_args()

def load_checkpoint(checkpoint_name):
    import os
    save_dir='./checkpoint/'
    #if checkpoint is not absolute path, add default path.
    if os.path.isfile(checkpoint_name):
        filepath=checkpoint_name
    else:
        filepath=save_dir+checkpoint_name
    checkpoint = torch.load(filepath)

    # The type of architecture is being used for the loaded checkpoint
    model_name = checkpoint['architecture']

    # Load the appropriate pre-trained model
    model = load_pretrained_model(model_name)

    # Assign values from the checkpoint to the model
    model.class_to_idx = checkpoint['image_dataset']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    return  model


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    image.thumbnail([255,255])
    image=img_center_crop(image)
    np_image = np.array(image)/255.0
    means = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (np_image - means)/ std
    img = normalized.transpose((2,0,1))
    #print(img.shape)
    return img

def img_center_crop(img):
    # Crop
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    return img.crop((left_margin, bottom_margin, right_margin, top_margin))

def imshow(image, ax=None, title=None):
    if ax is None:
        _, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax
def predict(processed, model, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    Input:
      image_path
      model
      topk: top k
    return
      probs: probilities
      classes: class
    '''
    # TODO: Implement the code to predict the class from an image file
    # Test out your network!
    model.eval()
    import pdb
    #pdb.set_trace()

    img = Variable(torch.from_numpy(processed))
    # in persion 0, increase a dimension.
    img = img.unsqueeze(0)
    device=set_device(model, gpu)
    img = img.type_as(torch.FloatTensor()).to(device)
    # Calculate the class probabilities (softmax) for img
    output = model.forward(img)
    #output.cpu()  Use Tensor.cpu() to copy the tensor to host memory first. can't convert CUDA tensor to numpy.
    ouput = torch.exp(output.cpu())
    probs, classes=ouput.topk(topk)
    # convert tensor to numpy , then to list
    probs = probs.data.numpy().tolist()[0]
    classes = classes.data.numpy().tolist()[0]
    # convert classes to class indices
    classes_toidx={idx:oid for oid,idx in model.class_to_idx.items()}
    classes_idx=[classes_toidx[i] for i in classes]
    return probs, classes_idx


def display_predict(probs, classeidx, category_names):
    '''
    Display an image along with the top 5 classes
    '''
    # Display probabilities and Classes
    y_labels =label_map( category_names, classeidx)
    print (" Classes:  Probabilie  ")
    for c, p in zip(y_labels, probs):
        print("{}  : {:.3%}".format(c,p))
def main():
    in_args=predict_args()
    model = load_checkpoint(in_args.checkpoint)
    processed = process_image(in_args.image_path)
    probs, classes = predict(processed, model, in_args.top_k, in_args.gpu)
    display_predict(probs, classes, in_args.category_names)

if __name__ == '__main__':
    main()
