import os
import time
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import torch.optim as optim 
from torch.optim import lr_scheduler
from datasets import get_train_valid_loader
from ResultWriter import ResultWriter
from statistics import accuracy, AverageMeter, ProgressMeter
from torchsummary import summary

from PIL import Image
import matplotlib.pyplot as plt

def train(args, model, dataloader, loader_len, criterion, optimizer, scheduler, use_gpu, epoch, save_file_name='train.csv'):
    
    ### Local Parameters
    resultWriter = ResultWriter(args.save_path, save_file_name)
    if epoch == 0:
        resultWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5', 'lr'])

    # use gpu or not
    device = torch.device('cuda' if use_gpu else 'cpu')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    
    # update lr here if using stepLR
    scheduler.step(epoch)

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

        # measure accuracy and record loss
        acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))

        #Reseting Gradients
        optimizer.zero_grad()

        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))
        
        # constant lr now

        #Backward
        loss.backward()
        optimizer.step()
        
        # epoch_loss = running_loss / len(train_size)
        # epoch_acc = running_corrects / len(train_size) * 100.
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            progress.display(i)
    
    resultWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item(), scheduler.optimizer.param_groups[0]['lr']])
    # print('[Train #{}] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))
    print('lr:%.6f' % scheduler.optimizer.param_groups[0]['lr'])
    print('Train ***    Loss:{losses.avg:.2e}    Acc@1:{top1.avg:.2f}    Acc@5:{top5.avg:.2f}'.format(losses=losses, top1=top1, top5=top5))
    
    if epoch % args.save_epoch_freq == 0 and epoch != 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

def validate(args, model, dataloader, loader_len, criterion, use_gpu, epoch, ema=None, save_file_name='val.csv'):
    '''
    validate the model
    '''

    # save result every epoch
    resultWriter = ResultWriter(args.save_path, save_file_name)
    if epoch == 0:
        resultWriter.create_csv(['epoch', 'loss', 'top-1', 'top-5'])

    device = torch.device('cuda' if use_gpu else 'cpu')

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        loader_len,
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()

    end = time.time()

    # Iterate over data
    for i, (inputs, labels) in enumerate(dataloader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0], inputs.size(0))
            top5.update(acc5[0], inputs.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            

    # if args.ema_decay > 0:
    #     # restore the origin parameters after val
    #     ema.restore()
    # write val result to file
    resultWriter.write_csv([epoch, losses.avg, top1.avg.item(), top5.avg.item()])

    print(' Val  ***    Loss:{losses.avg:.2e}    Acc@1:{top1.avg:.2f}    Acc@5:{top5.avg:.2f}'.format(losses=losses, top1=top1, top5=top5))
    
    if epoch % args.save_epoch_freq == 0 and epoch != 0:
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        torch.save(model.state_dict(), os.path.join(args.save_path, "epoch_" + str(epoch) + ".pth"))

    top1_acc = top1.avg.item()
    top5_acc = top5.avg.item()
    
    return top1_acc, top5_acc

def train_model(args, model, dataloader, loaders_len, criterion, optimizer, scheduler, use_gpu):
    '''
    train the model
    '''
    since = time.time()

    # ema = None
    # exponential moving average
    # if args.ema_decay > 0:
    #     ema = EMA(model, decay=args.ema_decay)
    #     ema.register()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    correspond_top5 = 0.0

    for epoch in range(args.start_epoch, args.num_epochs):

        epoch_time = time.time()
        train(args, model, dataloader['train'], loaders_len['train'], criterion, optimizer, scheduler, use_gpu, epoch)
        top1_acc, top5_acc  = validate(args, model, dataloader['val'], loaders_len['val'], criterion, use_gpu, epoch)
        # test(args, model, testloader, test_loader_len, criterion, use_gpu, epoch)
        epoch_time = time.time() - epoch_time
        print('Time of epoch-[{:d}/{:d}] : {:.0f}h {:.0f}m {:.0f}s\n'.format(epoch, args.num_epochs, epoch_time // 3600, (epoch_time % 3600) // 60, epoch_time % 60))

        # deep copy the model if it has higher top-1 accuracy
        if top1_acc > best_acc:
            best_acc = top1_acc
            correspond_top5 = top5_acc
            # if args.ema_decay > 0:
            #     ema.apply_shadow()
            best_model_wts = copy.deepcopy(model.state_dict())
            # if args.ema_decay > 0:
            #     ema.restore()

    print(os.path.split(args.save_path)[-1])
    print('Best val top-1 Accuracy: {:4f}'.format(best_acc))
    print('Corresponding top-5 Accuracy: {:4f}'.format(correspond_top5))
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed % 3600) // 60, time_elapsed % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save best model weights
    if args.save:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model_wts-' + '{:.2f}'.format(best_acc) + '.pth'))
    return model


if __name__ == "__main__":
    # Construct the argument parser.
    parser = argparse.ArgumentParser(description='PyTorch implementation of MobileNetV3')
    # Root catalog of images
    # parser.add_argument('--data-dir', type=str, default='/media/data2/chenjiarong/ImageData')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num-workers', type=int, default=4)
    #parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--print-freq', type=int, default=1000)
    parser.add_argument('--save-epoch-freq', type=int, default=1)
    parser.add_argument('--save-path', type=str, default='../outputs/trained_model')
    parser.add_argument('-save', default=False, action='store_true', help='save model or not')
    parser.add_argument('--resume', type=str, default='', help='For training from one checkpoint')
    parser.add_argument('--start-epoch', type=int, default=0, help='Corresponding to the epoch of resume')
    parser.add_argument('--ema-decay', type=float, default=0.9999, help='The decay of exponential moving average ')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='The dataset to be trained')
    parser.add_argument('--width-multiplier', type=float, default=1.0, help='width multiplier')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr-decay', type=str, default='step', help='learning rate decay method, step, cos or sgdr')
    parser.add_argument('--step-size', type=int, default=3, help='step size in stepLR()')
    parser.add_argument('--gamma', type=float, default=0.99, help='gamma in stepLR()')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--bn-momentum', type=float, default=0.1, help='momentum in BatchNorm2d')
    parser.add_argument('-use-seed', default=False, action='store_true', help='using fixed random seed or not')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('-deterministic', default=True, action='store_true', help='torch.backends.cudnn.deterministic')
    parser.add_argument('-zero-gamma', default=False, action='store_true', help='zero gamma in BatchNorm2d when init')
    args = parser.parse_args()

    TRAIN_DIR = '../datasets/train'
    VALID_DIR = '../datasets/val'
    TEST_DIR = '../datasets/test'
    save_path = '../output'

    #define train transform
    train_transform= transforms.Compose([
        transforms.Resize((64, 64)), # Require size by Resnet-18
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])

    valid_transform= transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                            (0.229, 0.224, 0.225))
    ])

    # use gpu or not
    use_gpu = torch.cuda.is_available()

    print("use_gpu:{}".format(use_gpu))
    if use_gpu:
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
        print('torch.backends.cudnn.deterministic:' + str(args.deterministic))
    
    train_loader, valid_loader, train_size, valid_size,\
    classes = get_train_valid_loader(TRAIN_DIR, 
                                    VALID_DIR,
                                    train_batch_size = 64,
                                    val_batch_size = 16,
                                    train_transform = train_transform,
                                    valid_transform = valid_transform,
                                    num_workers = 4,
                                    pin_memory = True)

    dataloaders = {'train' : train_loader, 'val' : valid_loader}
    loaders_len = {'train': train_size, 'val' : valid_size}

    print(f"[INFO]: Number of training images: {train_size}")
    print(f"[INFO]: Number of validation images: {valid_size}")
    print(f"[INFO]: Class names: {classes}\n")

    # Create model here
    # model = models.resnet18(pretrained = True)
    model = models.resnet18()
    # num_features = model.fc.in_features     #extract fc layers features
    # model.fc = nn.Linear(num_features, 2) #(num_of_class == 2)
    # model.fc = nn.Sequential(
    #     nn.Linear(2048, 1, bias = True),
    #     nn.Sigmoid()
    # )
    if use_gpu:
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(torch.device('cuda'))
    else:
        model.to(torch.device('cpu'))

    # summary(model, input_size = (3, 64, 64))
    #Loss Function
    criterion = nn.CrossEntropyLoss() 

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Learning Rate Scheduler
    lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = args.step_size, gamma = args.gamma)
    
    model = train_model(args=args,
                    model=model,
                    dataloader=dataloaders,
                    loaders_len=loaders_len,
                    criterion=criterion,
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    use_gpu=use_gpu)
