# Based on https://github.com/pytorch/examples/tree/master/imagenet
import os
import sys
import shutil
import random
import logging
import argparse
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from models import swin_transformer, i3d_head, vivit
from utils.utils import AverageMeter

WORK_PATH='log/face_pretrain_vivit'
DATA_ROOT='/home/yoon/data/face/VGG-Face2/face_extract'
BATCH_SIZE=1024
GPUS=[0,1,2,3,4,5,6,7]
LR=1e-3
NUM_EPOCH=50
MODEL='vivit'
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, num_class):
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter("Loss", ":.4e")
    model.train()
    pbar = tqdm(train_loader, desc="Train Iteration")
    for videos, targets in pbar:
        # Process input
        targets = targets.to("cuda", non_blocking=True)
        videos = videos.to('cuda', non_blocking=True)
        videos = videos.reshape(videos.shape[0], 1, videos.shape[1], videos.shape[2], videos.shape[3])
        #print('debug41', targets.shape)
        #targets = F.one_hot(targets, num_classes=num_class).to('cuda', non_blocking=True)
        #targets = targets.reshape(-1, 1).to('cuda', non_blocking=True)
        

        # Compute output
        preds = model(videos)
        # Loss
        #print('debug', preds.shape, targets.shape)
        loss = criterion(preds, targets)
        losses.update(loss.item(), videos.size(0))
        acc = accuracy(preds, targets)
        #acc = (preds == targets).float().sum()
        top1.update(acc[0], videos.size(0))

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description("loss %f, acc %f" % (losses.avg, top1.avg))
    return top1.avg.item(), losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, num_class):
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter("Loss", ":.4e")
    #all_pred = []
    #all_true = []
    #mse_loss_func = nn.MSELoss()

    # Switch to eval mode
    model.eval()
    #headmodel.eval()
    for videos, targets in tqdm(val_loader, desc="Val Iteration"):
        
        # Process input
        targets = targets.to("cuda", non_blocking=True)
        videos = videos.to('cuda', non_blocking=True)
        videos = videos.reshape(videos.shape[0], 1, videos.shape[1], videos.shape[2], videos.shape[3])
        #targets = targets.reshape(-1, 1).to('cuda', non_blocking=True)
        #targets = F.one_hot(targets, num_classes=num_class).to('cuda', non_blocking=True)
        # Compute output
        #embs = model(videos)
        #preds = headmodel(embs)
        preds = model(videos)
        #acc = (preds == targets).float().sum()
        acc = accuracy(preds, targets)
        top1.update(acc[0], videos.size(0))
        # Loss
        loss = criterion(preds, targets)
        losses.update(loss.item(), videos.size(0))

    return top1.avg.item(), losses.avg


def main():
    
    transform_dict = {
        'src': transforms.Compose(
        [transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'tar': transforms.Compose(
        [transforms.Resize((112, 112)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}

    datasets = {}
    datasets['train'] = ImageFolder(os.path.join(DATA_ROOT,'train'), transform=transform_dict['src'])
    datasets['test'] = ImageFolder(os.path.join(DATA_ROOT,'test'), transform=transform_dict['tar'])
    NUM_CLASS = len(datasets['train'].classes)
    trainloader = torch.utils.data.DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPUS), drop_last=True)
    testloader  = torch.utils.data.DataLoader(datasets['test'], batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPUS), drop_last=True)
    print("Number of Training Classes: {}".format(NUM_CLASS))

    if MODEL == 'vivit':
         model = vivit.ViViT(112, 16, NUM_CLASS, 1, heads=6, dim_head=128, depth=6).cuda()
    elif MODEL == 'swin':
        head_model = i3d_head.I3DHead(NUM_CLASS,1024)
        model = swin_transformer.SwinTransformer3D(head_model, patch_size=(2,4,4), drop_path_rate=0.2, depths=[2, 2, 18, 2],
                           embed_dim=128,
                           num_heads=[4, 8, 16, 32])
    
    model = nn.DataParallel(model, device_ids = GPUS)
    model = model.to('cuda')
    #head_model = head_model.to(device)
    #print(model, head_model)

    # Loss function
    #criterion = nn.L1Loss().to('cuda')
    #criterion = nn.SmoothL1Loss().to("cuda")
    criterion = nn.CrossEntropyLoss().to("cuda")                                                                                                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                         

    # Optimise only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    #optimiser = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    optimiser = optim.AdamW(parameters, lr=LR, betas=(0.9, 0.999), weight_decay=0.05)
    best_loss = 100
    for epoch in trange(NUM_EPOCH, desc="Epoch"):
        train_acc, train_loss = train(trainloader, model, criterion, optimiser, NUM_CLASS)
        #acc, val_loss = validate(testloader, model, criterion, NUM_CLASS)
        acc = train_acc
        val_loss = train_loss
        # Evaluate on validation set
 
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if is_best:
            state = {
                "epoch": epoch + 1,
                "state_dict": model.module.state_dict(), #ycyoon
                "optimiser": optimiser.state_dict(),
                "best_loss": best_loss,
            }
            path = os.path.join(WORK_PATH, "best_test_model.pth.tar")
            torch.save(state, path)

            print("\nModel saved at epoch {}".format(epoch + 1))
            logging.info("Model saved at epoch {}".format(epoch + 1))

        print(
            """Test Train Loss: {:.4f}, Test Val Loss/Best: {:.4f}/{:.4f}, 
			acc: {:.4f}""".format(
                train_loss, val_loss, best_loss, acc
            )
        )
        logging.info(
            "({}/{}) Test Train Loss: {:.4f}, Test Val Loss/Best: {:.4f}/{:.4f}, train_acc: {:.4f}, val_acc: {:.4f}".format(
                epoch + 1, NUM_EPOCH, train_loss, val_loss, best_loss, train_acc, acc
            )
        )

if __name__=="__main__":
    main()
