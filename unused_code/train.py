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
from ResultWriter import ResultWriter
from statistics import AverageMeter, ProgressMeter

from PIL import Image
import matplotlib.pyplot as plt


# ROOT_DIR = '../input/datasets'
# TRAIN_DIR = '../input/datasets/final_train'
# VAL_DIR = '../input/datasets/val'
# TEST_DIR  = '../input/datasets/test'

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

# def accuracy(preds, trues):
    
#     ### Converting preds to 0 or 1
#     preds = [1 if preds[i] >= 0.5 else 0 for i in range(len(preds))]
    
#     ### Calculating accuracy by comparing predictions with true labels
#     acc = [1 if preds[i] == trues[i] else 0 for i in range(len(preds))]
    
#     ### Summing over all correct predictions
#     acc = np.sum(acc) / len(preds)
    
#     return (acc * 100)

def train(model, dataloader, loader_len, criterion, optimizer, scheduler, use_gpu, epoch, save_path, save_file_name='train.csv'):
    
    ### Local Parameters
    resultWriter = ResultWriter(save_path, save_file_name)
    if epoch == 0:
        resultWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5', 'lr'])

    # use gpu or not
    device = torch.device('cuda' if use_gpu else 'cpu')
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Acc', ':.4e')
    progress = ProgressMeter(
    loader_len,
    [batch_time, data_time, losses, accuracy],
    prefix="Epoch: [{}]".format(epoch))
    
    # Set model to training mode
    model.train()

    end = time.time()

    ###Iterating over data loader
    for i, (inputs, labels) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)

        #Loading images and labels to device
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        #measure accuracy
        acc = torch.sum(preds == labels.data)
        #Reseting Gradients
        optimizer.zero_grad()

        losses.update(loss.item(), inputs.size(0))
        accuracy.update(acc, inputs.size(0))
        # running_loss += loss.item() * inputs.size(0)
        
        # constant lr now

        #Backward
        loss.backward()
        optimizer.step()
        
        # epoch_loss = running_loss / len(train_size)
        # epoch_acc = running_corrects / len(train_size) * 100.
        batch_time.update(time.time() - end)
        end = time.time()
        
        progress.display(i)
    
    resultWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item()])
    # print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
    print('Train ***    Loss:{losses.avg:.2e}    Acc@1:{accuracy.avg:.2f} '.format(losses=losses, accuracy=accuracy))
    if epoch != 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, "epoch_" + str(epoch) + ".pth"))

def val(valid_loader, best_val_acc, save_file_name='test.csv'):
    
    # Set model to evaluate mode
    model.eval()

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
    TRAIN_DIR = '../datasets/train'
    VAL_DIR = '../datasets/val'
    TEST_DIR = '../datasets/test'
    save_path = '../output'

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
        loss, acc, _time = train(model, train_loader, train_size, save_path)
        
        #Print Epoch Details
        print("\nTraining")
        print("Epoch {}".format(epoch+1))
        print("Loss : {}".format(round(loss, 4)))
        print("Acc : {}".format(round(acc, 4)))
        print("Time : {}".format(round(_time, 4)))
        
        ###Validation
        loss, acc, _time, best_val_acc = val(model, valid_loader, valid_size, best_val_acc)
        
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