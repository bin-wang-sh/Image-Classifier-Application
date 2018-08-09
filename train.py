import json
import numpy as np
import torch
from torch import optim,nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets,models,transforms
import argparse
from collections import OrderedDict
from utils import *
def trian_args():
    parser = argparse.ArgumentParser( description='Get Train network parameters' )
    parser.add_argument('data_dir', default='flowers', type=str,
                     help = "Data source directory")
    parser.add_argument('--save_dir',default="checkpoint",type=str,
                     help="save directory")
    parser.add_argument('--arch', default="vgg16",type=str, dest='model_name',
                     help="Select model: vgg19_bn, densenet121, vgg16")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                     help="Learning rate")
    parser.add_argument('--hidden_units', default=512, type=int,
                     help="Assignment hidden_units number")
    parser.add_argument('--epochs', default=5, type=int,
                     help="epoch number")
    parser.add_argument('--gpu', default=False, action='store_true',
                     help="use --gpu to enable GPU process")
    return parser.parse_args()


def train_network(model, dataloaders, validloaders,  epochs, print_every,learning_rate=0.001, gpu=False):
    #define loss function
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    steps=0
    #enable dropout in trainning dataset:
    model.train()
    # if GPU is available and GPU is set , then use GPU , or use CPU.
    device = set_device(model, gpu)
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs,labels) in enumerate(dataloaders):
            steps += 1
            #use Variable to wrap the tensor and enable gradient descent.
            inputs, labels = Variable(inputs), Variable(labels)
            #move tensor to target device(GPU|CPU)
            inputs, labels = inputs.to(device), labels.to(device)

            #optimizer gradient to zero
            optimizer.zero_grad()
            #forward and backwards
            outputs=model.forward(inputs)
            loss=criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0 :
                valid_loss,accuracy = validate_nn(model,validloaders,criterion,gpu)
                print("Epoch {} / {}..".format(e+1,epochs),
                     "loss:  {:.4f}".format(running_loss/print_every),
                     "Validateion Loss: {:.3f}.. ".format(valid_loss),
                     "Validation Accuracy: {:.3f}".format(accuracy)
                     )

                running_loss=0

def validate_nn(model, testloader, criterion,gpu=False):
    # Model in inference mode, dropout is off
    model.eval()
    device = set_device(model, gpu)
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        for ii, (images, labels) in enumerate(testloader):
            # Set volatile to True so we don't save the history
            inputs = Variable(images)
            labels = Variable(labels)
            inputs, labels = inputs.to(device), labels.to(device)
            output = model.forward(inputs)
            loss = criterion(output, labels)
            test_loss += loss.item()

            ## Calculating the accuracy
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output).data
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()

    # Make sure dropout is on for training
    model.train()
    return test_loss/len(testloader), accuracy/len(testloader)

#--------------main()---------------------

def transform_data(data_dir='flowers'):
    '''
      parameters:
         data_dir: input data directory ,default is 'flowers'
      return:
         dataloaders : train  data  Loader
         validloaders: validate data loader
         testloaders: test data loader
    '''
    #Define data source.
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # TODO: Define your transforms for the training, validation, and testing sets
    data_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])
    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform = data_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size=32)
    return  image_datasets,datasloaders, validloaders,  testloaders

def replace_classifier(model, model_name, hidden_units=4096):
    ''' Create a new classifier to replace one in  pretrained model.
    parameters:
        model ,Torchvision.Models, - Pretrained model
        model_name, str,  - Model architecture
        hidden_units, int, - Desired number of nodes in hidden layer
    return:
        model ,models,  - Pretrained model with new classifier
    '''
    #model_input_units={'vgg19_bn':25088, 'densenet121':1024, 'vgg16':25088}
    #input_units=model_input_units[model_name]
    input_units = model.classifier[0].in_features
    new_classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_units, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(p=0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = new_classifier
    return model
def check_accuracy_on_test(model,testloader, gpu=False):
    correct = 0
    total = 0
    #disable dropout in trainning dataset:
    model.eval()
    # if GPU is available use GPU , or use CPU.
    device = set_device(model, gpu)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            #move tensor to target device(GPU|CPU)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    #enable dropout in trainning dataset:
    model.train()
    print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))

def save_checkpoint(model, model_name,image_datasets,save_dir):
    filepath=save_dir+'/'+model_name+'.pth'
    input_size = model.classifier[0].in_features
    model.class_to_idx = image_datasets.class_to_idx
    checkpoint = {
        'architecture': model_name,
        'input_size': input_size,
        'output_size': 102,
        'image_dataset': model.class_to_idx,
        'classifier': model.classifier,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, filepath)
#========training data ==================
def main():
    in_args = trian_args()
    train_datasets, trainloaders, validloaders,  testloaders=transform_data(in_args.data_dir)
    model = load_pretrained_model(in_args.model_name)
    model = replace_classifier(model, in_args.model_name, in_args.hidden_units)

    train_network(model, trainloaders, validloaders, in_args.epochs, 20,in_args.learning_rate, in_args.gpu)
    check_accuracy_on_test(model, testloaders, in_args.gpu)
    save_checkpoint(model, in_args.model_name, train_datasets, in_args.save_dir)
if __name__ == '__main__':
    main()
