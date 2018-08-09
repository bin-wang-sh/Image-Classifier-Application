import json
import torch
from torch import optim,nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,models,transforms

def label_map(category_names,classidx):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    classes_name=[cat_to_name[i] for i in classidx]
    return classes_name

def load_pretrained_model(model_name='vgg16'):
    ''' Load a pretrained model based on user input.
    parameters:
        model_name, str,  - target network architecture.
    return:
        model, models, - Pretrained model.
    '''
    # Select models
    if model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'vgg16': # VGG16 (Default)
        model = models.vgg16(pretrained=True)
    elif  model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    return model

def set_device(model, gpu=False):
    if gpu :
        if torch.cuda.is_available():
            model.cuda()
            device='cuda'
        else:
            print('GPU is not available, CPU is used')
            model.cpu()
            device='cpu'
    else:
        model.cpu()
        device='cpu'
    return device
