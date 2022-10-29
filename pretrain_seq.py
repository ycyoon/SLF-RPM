'''
Random sequence prediction self-training learning
'''
import os
import sys
import shutil
import random
import logging
import argparse
import math
from turtle import st

from tqdm import tqdm, trange

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.dataset import FERAugDataset
from utils.utils import accuracy, AverageMeter
from utils.augmentation import Transformer
from models import swin_transformer, i3d_head, vivit

#RETRAIN='log/seq_pretrain/best_train_model_at_176.pth.tar'
RETRAIN=False
WORK_PATH='log/seq_pretrain/'
DATA_ROOT='/home/yoon/data/fer/self-supervised-learning/faces/'
PRETRAIN_VGG='log/face_pretrain_vivit/best_test_model.pth.tar'
GPUS=[0,1,2,3,4,5,6,7]
NUM_EPOCH=600
LR0=1e-3
LR1=4e-3
BATCH_SIZE=32
SEQSIZE=150
STRIDE=2
MASKNUM=3
MODELNAME='vivit'
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def main():
    writer = SummaryWriter(WORK_PATH)

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    cudnn.deterministic = True

    print("Use GPU: {} for training".format(GPUS))
    device = torch.device("cuda")     
    #numclass = math.factorial(MASKNUM) #가능한 모든 경우의 수
    numclass = 2
    print('Index pred Numm class', numclass)
    

    if MODELNAME == "swin":
        head_model_random_shift = i3d_head.I3DHead(numclass,1024)
        head_model_histogram = i3d_head.I3DHead(256,1024)
        
        model = swin_transformer.SwinTransformer3D([head_model_random_shift, head_model_histogram], patch_size=(2,4,4), drop_path_rate=0.2, depths=[2, 2, 18, 2],
                            embed_dim=128,
                            num_heads=[4, 8, 16, 32])

        if RETRAIN:
            print("=> Loading checkpoint '{}'".format(RETRAIN))
            checkpoint = torch.load(RETRAIN, map_location="cpu")
            face_state_dict = checkpoint["state_dict"]                       
            for k in list(face_state_dict.keys()):                
                # remove module prefix
                if k.startswith("module."):
                    face_state_dict[k[7:]] = face_state_dict[k]
                    del face_state_dict[k]
            model.load_state_dict(face_state_dict, strict=True)      
            print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"])) 
        # Load from pretrained model
        elif PRETRAIN_VGG:
            if os.path.isfile(PRETRAIN_VGG):
                print("=> Loading checkpoint '{}'".format(PRETRAIN_VGG))
                checkpoint = torch.load(PRETRAIN_VGG, map_location="cpu")
                face_state_dict = checkpoint["state_dict"]       
                state_dict = {}
                for k in list(face_state_dict.keys()):                
                    if "fc_cls" not in k:
                        state_dict[k] = face_state_dict[k]                    

                model.load_state_dict(state_dict)            
                print("=> Loaded pre-trained model '{}'".format(PRETRAIN_VGG))
                print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"]))
        
        
    elif MODELNAME== "vivit":
        dim = 192
        
        model = vivit.ViViT(112, 16, numclass, SEQSIZE//STRIDE, heads=6, dim=dim, dim_head=128, depth=6, hist=256*3*5).cuda()
        # Load from pretrained model
        if RETRAIN:
            print("=> Loading checkpoint '{}'".format(RETRAIN))
            checkpoint = torch.load(RETRAIN, map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=True)      
            print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"])) 
        # Load from pretrained model
        elif PRETRAIN_VGG:
            if os.path.isfile(PRETRAIN_VGG):
                print("=> Loading checkpoint '{}'".format(PRETRAIN_VGG))
                checkpoint = torch.load(PRETRAIN_VGG, map_location="cpu")
                face_state_dict = checkpoint["state_dict"]       
                state_dict = {}
                for k in list(model.state_dict().keys()):                
                    if "mlp_head" not in k and "pos_embedding" not in k:
                        state_dict[k] = face_state_dict[k]
                    else:
                        state_dict[k] = model.state_dict()[k]
                    
                model.load_state_dict(state_dict, strict=True)            
                print("=> Loaded pre-trained model '{}'".format(PRETRAIN_VGG))
                print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"]))       

    model = nn.DataParallel(model, device_ids = GPUS)
    model = model.to(device)
    # Loss function
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = nn.SmoothL1Loss().to(device)

    # Optimiser function
    #optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimiser1 = optim.AdamW(parameters, lr=LR0, betas=(0.9, 0.999), weight_decay=0.05)
    optimiser2 = optim.AdamW(parameters, lr=LR1, betas=(0.9, 0.999), weight_decay=0.05)
    best_state = {}
    # Load data
    trs = Transformer(None, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    
    train_dataset = FERAugDataset(
        os.path.join(DATA_ROOT, 'train'),
        masknum=MASKNUM,
        transforms=trs,
        vid_frame=SEQSIZE,
        vid_frame_stride=STRIDE
    )
    val_dataset = FERAugDataset(
        os.path.join(DATA_ROOT, 'test'),
        masknum=MASKNUM,
        transforms=trs,
        vid_frame=SEQSIZE,
        vid_frame_stride=STRIDE
    )

    best_loss = sys.maxsize
    best_top1 = 0

    train_sampler = None
    
    is_best = False
    # Train model
    for epoch in trange(NUM_EPOCH, desc="Epoch"):
        if epoch % 10 == 0:
            task = 0 if random.random() > 0.5 else 1
            model.hist = task
            train_dataset.task = task
            val_dataset.task = task
            model.hist = task
            train_loader = DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=(train_sampler is None),
                sampler=train_sampler,
                num_workers=8,
                pin_memory=True,
                drop_last=True,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
                drop_last=False,
            )
        if task == 0:            
            train_top1, train_loss  = train(train_loader, model, criterion1, optimiser1, task)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Accuracy/train", train_top1, epoch)
            top1, val_loss  = validate(val_loader, model, criterion1, task)
            writer.add_scalar("Loss/valid", val_loss, epoch)
            writer.add_scalar("Accuracy/valid", top1, epoch)
            best_top1 = max(top1, best_top1)
        else:
            train_loss  = train(train_loader, model, criterion2, optimiser2, task)
            writer.add_scalar("Loss/train", train_loss, epoch)
            val_loss  = validate(val_loader, model, criterion2, task)
            writer.add_scalar("Loss/valid", val_loss, epoch)
            is_best = val_loss <= best_loss
            best_loss = min(val_loss, best_loss)
        
            if is_best:
                best_state = {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict(),
                    "optimiser1": optimiser1.state_dict(),
                    "optimiser2": optimiser2.state_dict(),
                    "best_loss": best_loss,
                    "best_top1": best_top1,
                }
                
                logging.info("Model saved at epoch {}".format(epoch + 1))
                print("\nModel saved at epoch {}".format(epoch + 1))
        
        if task == 0:
            print("Shuffling Val Loss/Best: {:.4f}/{:.4f}, Val Acc/Best: {:.4f}/{:.4f}".format(val_loss, best_loss, top1, best_top1))
        else:
            print("Histogram Val Loss/Best: {:.4f}/{:.4f}".format(val_loss, best_loss))        

        if epoch % 30 == 0:
            path = os.path.join(WORK_PATH, "best_train_model_at_%d.pth.tar" % (best_state['epoch']))           
            torch.save(best_state, path)


    writer.close()


def train(train_loader, model, criterion, optimizer, task):
    top1 = AverageMeter("Acc@1", ":6.2f")
    losses = AverageMeter("Loss", ":.4e")   

    model.train()
    pbar = tqdm(train_loader, desc="Train Iteration")            
    for videos, targets in pbar:
        
        targets = targets.to("cuda", non_blocking=True)
        videos = videos.to('cuda', non_blocking=True)

        # Compute output
        preds = model(videos)        
        # Contrastive loss
        loss = criterion(preds, targets)
        losses.update(loss.item(), videos.size(0))
                
        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if task == 0:
            acc = accuracy(preds, targets)
            top1.update(acc[0], videos.size(0))
            pbar.set_description("shuffle loss %f acc %f" % (losses.avg, top1.avg))
        else:
            pbar.set_description("histogram loss %f" % (losses.avg))
    if task == 0:
        return top1.avg.item(), losses.avg
    else:
        return losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, task):
    top1 = AverageMeter('Acc@1', ':6.2f')
    losses = AverageMeter("Loss", ":.4e")

    # Switch to eval mode
    model.eval()
    pbar = tqdm(val_loader, desc="Val Iteration")
    for videos, targets in pbar:
        
        # Process input
        targets = targets.to("cuda", non_blocking=True)
        videos = videos.to('cuda', non_blocking=True)

        preds = model(videos)
        # Loss
        loss = criterion(preds, targets)
        losses.update(loss.item(), videos.size(0))

        if task == 0:
            acc = accuracy(preds, targets)
            top1.update(acc[0], videos.size(0))
            pbar.set_description("shuffle loss %f), acc %f" % (losses.avg, top1.avg))
        else:
            pbar.set_description("histpred %f(dt) vs %f(gt)  (loss %f)" % (preds[0][5], targets[0][5], losses.avg))
        
    if task == 0:
        return top1.avg.item(), losses.avg
    else:
        return losses.avg

if __name__ == "__main__":
    main()
