import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import torch.optim as optim 

from PIL import Image
import matplotlib.pyplot as plt


# ROOT_DIR = '../input/datasets'
# TRAIN_DIR = '../input/datasets/final_train'
# VAL_DIR = '../input/datasets/val'
# TEST_DIR  = '../input/datasets/test'

TRAIN_DIR = '../input/datasets/train'
VAL_DIR = '../input/datasets/val'
TEST_DIR = '../input/datasets/test'

# image is in jpg format
# cat0.jpg dog.0.jpg

# dogs_list = os.listdir(os.path.join(TRAIN_DIR,"dog"))
# cats_list = os.listdir(os.path.join(TRAIN_DIR,"cat"))
# dogs_val = os.listdir(os.path.join(VAL_DIR,"dog"))
# cats_val = os.listdir(os.path.join(VAL_DIR,"cat"))
# test_imgs = os.listdir(TEST_DIR)

# Concat two lists to form train imgs
# train_imgs =  dogs_list + cats_list
# val_imgs = dogs_val + cats_val

# train_imgs = os.listdir(TRAIN_DIR)
# val_imgs = os.listdir(VAL_DIR)
# test_imgs = os.listdir(TEST_DIR)

# Image labels
# class_to_int = {"dog" : 0, "cat" : 1}
# int_to_class = {0 : "dog", 1 : "cat"}

# transformation (scaling, rotation & flipping)
train_transform= transforms.Compose([
    transforms.Resize((224, 224)), # Require size by Resnet-18
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])

valid_transform= transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225))
])

def accuracy(preds, trues):
    
    ### Converting preds to 0 or 1
    preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    
    ### Calculating accuracy by comparing predictions with true labels
    acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    
    ### Summing over all correct predictions
    acc = np.sum(acc) / len(preds)
    
    return (acc * 100)

def train_one_epoch(train_loader):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for i, (inputs, labels) in enumerate(train_loader):
        
        #Loading images and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        #Reseting Gradients
        optimizer.zero_grad()

        #Forward
        outputs = model(inputs)
        # preds = model(images)
        _, preds = torch.max(outputs, 1)
        
        #Calculating Loss
        # _loss = criterion(preds, labels)
        loss = criterion(outputs, labels)
        # loss = _loss.item()
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
        
        #Backward
        loss.backward()
        optimizer.step()
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    train_logs["loss"].append(epoch_loss)
    train_logs["accuracy"].append(epoch_acc)
    train_logs["time"].append(total_time)
        
    return epoch_loss, epoch_acc, total_time

def val_one_epoch(valid_loader, best_val_acc):
    
    ### Local Parameters
    epoch_loss = []
    epoch_acc = []
    start_time = time.time()
    
    ###Iterating over data loader
    for i, (inputs, labels) in enumerate(valid_loader):
        
        #Loading images and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        #Forward
        # preds = model(inputs)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        
        #Calculating Loss
        loss = criterion(outputs, labels)
        epoch_loss.append(loss)
        
        #Calculating Accuracy
        acc = accuracy(preds, labels)
        epoch_acc.append(acc)
    
    ###Overall Epoch Results
    end_time = time.time()
    total_time = end_time - start_time
    
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    epoch_acc = np.mean(epoch_acc)
    
    ###Storing results to logs
    val_logs["loss"].append(epoch_loss)
    val_logs["accuracy"].append(epoch_acc)
    val_logs["time"].append(total_time)
    
    ###Saving best model
    if epoch_acc > best_val_acc:
        best_val_acc = epoch_acc
        torch.save(model.state_dict(),"resnet18_best.pth")
        
    return epoch_loss, epoch_acc, total_time, best_val_acc


if __name__ == "__main__":
    train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    classes = train_dataset.classes
    print(classes)
    train_size = len(train_dataset)
    print("Train size: ", train_size)

    valid_dataset = datasets.ImageFolder(root=VAL_DIR, transform=valid_transform)
    valid_size = len(valid_dataset)
    print("Valid size: ", valid_size)

    # Check GPU availability
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # data loader
    train_loader = DataLoader(
        train_dataset,batch_size=64,num_workers=4,pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset,batch_size=16,num_workers=4,pin_memory=True
    )

    dataloaders = {'train' : train_loader, 'val' : valid_loader}
    loaders_len = {'train': train_size, 'val' : valid_size}

    # # Create model here
    # model = models.resnet18(pretrained=False)
    model = models.resnet18(pretrained = True)
    num_features = model.fc.in_features     #extract fc layers features
    model.fc = nn.Linear(num_features, 2) #(num_of_class == 2)
    # num_features = model.fc.in_features     #extract fc layers features
    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 1, bias = True),
    #     nn.Sigmoid()
    # )

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Learning Rate Scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.5)

    #Loss Function
    criterion = nn.CrossEntropyLoss()  #(set loss function)

    # Logs - Helpful for plotting after training finishes
    train_logs = {"loss" : [], "accuracy" : [], "time" : []}
    val_logs = {"loss" : [], "accuracy" : [], "time" : []}

    # Loading model to device
    model.to(device)

    # No of epochs 
    epochs = 5

    best_val_acc = 0

    # train_one_epoch(train_loader)
    for epoch in range(epochs):
        
        ###Training
        loss, acc, _time = train_one_epoch(train_loader)
        
        #Print Epoch Details
        print("\nTraining")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))
        
        ###Validation
        loss, acc, _time, best_val_acc = val_one_epoch(valid_loader, best_val_acc)
        
        #Print Epoch Details
        print("\nValidating")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))

    #Plotting results
    #Loss
    plt.title("Loss")
    plt.plot(np.arange(1, 11, 1), train_logs["loss"], color = 'blue')
    plt.plot(np.arange(1, 11, 1), val_logs["loss"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    #Accuracy
    plt.title("Accuracy")
    plt.plot(np.arange(1, 11, 1), train_logs["accuracy"], color = 'blue')
    plt.plot(np.arange(1, 11, 1), val_logs["accuracy"], color = 'yellow')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()