from email.policy import strict
import os
import sys
import random
import shutil
import logging
import argparse
import traceback
from tqdm import tqdm, trange

import numpy as np

from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.backends import cudnn
from torch.utils.data import DataLoader

#from models import i3d_head, swin_transformer
from models import swin_transformer, i3d_head, vivit, TS_CAN
from utils.dataset import rPPGDataset
from utils.utils import AverageMeter
from utils.augmentation import  Transformer, Resize, StdAndNormalise
from torchvision import transforms

parser = argparse.ArgumentParser()

# Training setting
parser.add_argument(
    "--seed", default=None, type=int, help="seed for initializing training. "
)
parser.add_argument(
    "--epochs", default=50, type=int, help="number of total epochs to run"
)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
parser.add_argument("--dropout", default=0.0, type=float, help="ResNet dropout value")

# Test setting
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--scratch", action="store_true")
parser.add_argument(
    "--pretrained", default="", type=str, help="path to pretrained checkpoint"
)
parser.add_argument(
    "--retrain", default="", type=str, help="path to pretrained checkpoint"
)

# Data setting
parser.add_argument("--trainval_dataset_name", default="ubfc2", type=str)
parser.add_argument("--test_dataset_name", default="none", type=str)

parser.add_argument("--dataset_dir", default=None, type=str)
parser.add_argument(
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 1)",
)
parser.add_argument("--vid_frame", default=150, type=int)
parser.add_argument("--vid_frame_stride", default=2, type=int)

# Log setting
parser.add_argument("--log_dir", default="./path/to/your/log", type=str)
parser.add_argument("--run_tag", nargs="+", default=None)
parser.add_argument("--run_name", default=None, type=str)
parser.add_argument("--finetune", default="all", type=str)
parser.add_argument("--model_name", default="vivit", type=str)
args = parser.parse_args()
GPU_NUM = 8
MODEL_IMGSIZE = 72
def main():

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logging.basicConfig(
        filename=os.path.join(args.log_dir, "test_output.log"),
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.DEBUG,
    )

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logging.info(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    else:
        cudnn.benchmark = True


    # Simply call main_worker function
    try:
        main_worker(args)
    except Exception as e:

        logging.critical(e, exc_info=True)
        print(traceback.format_exc())


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

def main_worker(args):
    best_loss = sys.maxsize 
    device = torch.device("cuda")
    if args.model_name == "swin":
        head_model = i3d_head.I3DHead(args.vid_frame//args.vid_frame_stride,768)
        model = swin_transformer.SwinTransformer3D(head_model, patch_size=(4,4,4), drop_path_rate=0.2, depths=[2, 2, 6, 2],
                            embed_dim=96,
                            num_heads=[3, 6, 12, 24], in_chans=6)
        criterion = nn.MSELoss().to(device)
        # Load from pretrained model
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print("=> Loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                face_state_dict = checkpoint["state_dict"]       
                state_dict = {}
                for k in list(model.state_dict().keys()):   
                    mk = "encoder_q." + k
                    if mk in face_state_dict.keys():
                        state_dict[k] = face_state_dict[mk]   
                    else:
                        state_dict[k] = model.state_dict()[k]

                model.load_state_dict(state_dict)            
                print("=> Loaded pre-trained model '{}'".format(args.pretrained))
                #print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"]))
        
        if args.finetune == "head":
            # Freeze all layers but the last fc
            for name, param in model.named_parameters():
                if 'fc_cls' not in name:
                    param.requires_grad = False
            logging.info("=> Finetune only head layer")
        
        model.head = head_model    
    elif args.model_name == "vivit":
        #model = vivit.ViViT(112, 8, 1, 75, pool='mean', scale_dim=8, heads=6, depth=8, dropout=0.3, emb_dropout=0.3).cuda()
        model = vivit.ViViT(MODEL_IMGSIZE, 8, args.vid_frame//args.vid_frame_stride, args.vid_frame//args.vid_frame_stride, heads=2, dim_head=64, depth=2).cuda()
        criterion = nn.L1Loss().to(device)
        # Load from pretrained model
        #print(model)
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print("=> Loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                face_state_dict = checkpoint["state_dict"]
                #model.load_state_dict(face_state_dict, strict=True)   
                state_dict = {}
                #print(face_state_dict['encoder_q.space_token'])
                for k in list(model.state_dict().keys()):                
                    if "mlp_head" not in k:
                        state_dict[k] = face_state_dict['encoder_q.'+k]
                    elif "mlp_head2" not in k:
                        state_dict[k] = model.state_dict()[k]
                    
                model.load_state_dict(state_dict, strict=True)            
                print("=> Loaded pre-trained model '{}'".format(args.pretrained))
                #print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"]))
            else:
                print("=> No checkpoint found at '{}'".format(args.pretrained))
        elif args.retrain:
                print("=> Loading checkpoint '{}'".format(args.retrain))
                checkpoint = torch.load(args.retrain, map_location="cpu")
                face_state_dict = checkpoint["state_dict"]                       
                for k in list(face_state_dict.keys()):                
                    # remove module prefix
                    if k.startswith("module."):
                        face_state_dict[k[7:]] = face_state_dict[k]
                        del face_state_dict[k]
                model.load_state_dict(face_state_dict, strict=True)      
                print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"]))   
        if args.finetune == "head":
            # Freeze all layers but the last fc
            for name, param in model.named_parameters():
                if 'mlp_head' not in name:
                    param.requires_grad = False
            logging.info("=> Finetune only head layer")   
        
    elif args.model_name == "tscan":
        model = TS_CAN.TSCAN(frame_depth=args.vid_frame//args.vid_frame_stride, img_size=MODEL_IMGSIZE).to(device)
        criterion = torch.nn.MSELoss().to(device)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    model = nn.DataParallel(model)
    model = model.to('cuda')

    # Optimise only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimiser = optim.AdamW(parameters, lr=args.lr, weight_decay=0)

    # scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.95)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimiser,
                                        lr_lambda=lambda epoch: 0.99 ** epoch)
    # Load data
    augmentation = [Resize(MODEL_IMGSIZE), StdAndNormalise()]

    augmentation = transforms.Compose(
                [*augmentation]
            )
    
    train_dataset = rPPGDataset(args.trainval_dataset_name, args.dataset_dir, 1, augmentation, args.vid_frame, args.vid_frame_stride)
    val_dataset = rPPGDataset(args.trainval_dataset_name, args.dataset_dir, 0, augmentation, args.vid_frame, args.vid_frame_stride)
    
    train_sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        sampler=train_sampler,
        num_workers=args.workers,   
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )
    if args.test_dataset_name != "none":
        test_dataset = rPPGDataset(args.test_dataset_name, args.dataset_dir, -1, augmentation, args.vid_frame, args.vid_frame_stride)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=False,
        )
    
    #TODO: test 처리 함수 만들기 (MTTS-CAN 처럼 평균값으로 하기)      

    if args.evaluate:        
        mae, std, rmse, r = validate(val_loader, model, criterion, device)
        print(
            "Evaluation Result\n MAE: {:.4f}; SD: {:.4f}; RMSE: {:.4f}; R: {:.4f};".format(
                mae, std, rmse, r
            )
        )
        logging.info(
            "Evaluation Result\n MAE: {:.4f}; SD: {:.4f}; RMSE: {:.4f}; R: {:.4f};".format(
                mae, std, rmse, r
            )
        )
        return
    else:
        # Train model
        for epoch in trange(args.epochs, desc="Epoch"):
            baselen = GPU_NUM * args.vid_frame // args.vid_frame_stride
            train_loss = train(train_loader, model, criterion, optimiser, baselen, device)
            # scheduler.step()
            # Evaluate on validation set
            val_loss, std, rmse, r = validate(val_loader, model, criterion, device)        

            is_best = val_loss < best_loss
            best_loss = min(val_loss, best_loss)

            if is_best:
                state = {
                    "epoch": epoch + 1,
                    "state_dict": model.module.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "best_loss": best_loss,
                }
                path = os.path.join(args.log_dir, "best_test_model.pth.tar")
                torch.save(state, path)

                print("\nModel saved at epoch {}".format(epoch + 1))
                logging.info("Model saved at epoch {}".format(epoch + 1))
            #print learning rate
            for param_group in optimiser.param_groups:
                print("Learning rate: {}".format(param_group["lr"]))
                logging.info("Learning rate: {}".format(param_group["lr"]))

            print(
                """Test Train Loss: {:.4f}, Test Val Loss/Best: {:.4f}/{:.4f}, 
                Test SD: {:.4f}, Test RMSE: {:.4f}, Test R: {:.4f}""".format(
                    train_loss, val_loss, best_loss, std, rmse, r
                )
            )
            logging.info(
                "({}/{}) Test Train Loss: {:.4f}, Test Val Loss/Best: {:.4f}/{:.4f}, Test SD: {:.4f}, Test RMSE: {:.4f}, Test R: {:.4f}".format(
                    epoch + 1, args.epochs, train_loss, val_loss, best_loss, std, rmse, r
                )
            )

def train(train_loader, model, criterion, optimizer, base_len, device):
    losses = AverageMeter("Loss", ":.4e")
    model.train()
    batch = -1
    pbar = tqdm(train_loader, desc="Train Iteration")
    for data, labels in pbar:
        # Process input        
        if batch == -1:
            batch = data.size(0) 
        if data.size(0) < batch:
            continue
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.model_name == 'tscan':
            labels = labels.reshape(-1, 1)
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)
            data = data[:(N * D) // base_len * base_len]
            labels = labels[:(N * D) // base_len * base_len]
        preds = model(data)
        # Loss
        loss = criterion(preds, labels)
        
        # loss = 0.7*loss + 0.3*lossstd  #ycyoon 값이 몰리지 않도록 조정
        losses.update(loss.item(), data.size(0))

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description("loss %f" % (losses.avg))
    return losses.avg

@torch.no_grad()
def validate(val_loader, model, criterion, device):
    maes = AverageMeter("MAE", ":.4e")
    mses = AverageMeter("MSE", ":.4e")
    all_pred = []
    all_true = []
    mse_loss_func = nn.MSELoss()

    # Switch to eval mode
    model.eval()
    batch = -1
    pasum = 0; tasum = 0
    for data, labels in tqdm(val_loader, desc="Val Iteration"):
                # Process input
        if batch == -1:
            batch = data.size(0) 
        if data.size(0) < batch:
            continue
        data = data.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if args.model_name == 'tscan':
            labels = labels.reshape(-1, 1)
            N, D, C, H, W = data.shape
            data = data.view(N * D, C, H, W)
            labels = labels.view(-1, 1)

        preds = model(data)
        # Loss
        mae = criterion(preds, labels)
        mse = mse_loss_func(preds, labels)
        
        te = labels[-20:].tolist(); pe = preds[-20:].tolist()
        if args.model_name == 'tscan':
            te = [t[0] for t in te]; pe = [p[0] for p in pe] 
        ta = [round(t, 2) for t in te]; pa = [round(p, 2) for p in pe]
        tasum += np.std(ta); pasum += np.std(pa)

        maes.update(mae.item(), labels.size(0))
        mses.update(mse.item(), labels.size(0))

        all_pred.append(preds.detach().cpu())
        all_true.append(labels.detach().cpu())
    print('std of targets', tasum/len(val_loader), 'std of preds', pasum/len(val_loader))

    all_pred = torch.cat(all_pred).flatten()
    all_true = torch.cat(all_true).flatten()

    # Mean and Std
    diff = all_pred - all_true
    std = torch.std(diff)

    # MSE
    mse_loss = mses.avg

    # RMSE
    rmse_loss = np.sqrt(mse_loss)
    
    r, _ = pearsonr(all_true, all_pred)

    return maes.avg, std, rmse_loss, r


if __name__ == "__main__":
    main()
