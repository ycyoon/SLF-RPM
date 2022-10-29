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
from models import swin_transformer, i3d_head, vivit
from utils.dataset import MAHNOBHCIDataset, VIPLHRDataset, UBFC1Dataset, UBFC2Dataset, MergedDataset, PUREDataset, CohfaceDataset
from utils.utils import AverageMeter
from utils.augmentation import RandomROI, Transformer

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
parser.add_argument("--dataset_name", default="mahnob-hci", type=str)
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
parser.add_argument("--finetune", default="head", type=str)
parser.add_argument("--model_name", default="vivit", type=str)

def main():
    args = parser.parse_args()

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
        head_model = i3d_head.I3DHead(1,1024)
        model = swin_transformer.SwinTransformer3D(head_model, patch_size=(2,4,4), drop_path_rate=0.2, depths=[2, 2, 18, 2],
                            embed_dim=128,
                            num_heads=[4, 8, 16, 32])

        # Load from pretrained model
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print("=> Loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")
                face_state_dict = checkpoint["state_dict"]       
                state_dict = {}
                for k in list(face_state_dict.keys()):                
                    if "fc_cls" not in k:
                        state_dict[k] = face_state_dict[k]
                    else:
                        state_dict[k] = head_model.state_dict()[k.replace('head.','')]

                model.load_state_dict(state_dict)            
                print("=> Loaded pre-trained model '{}'".format(args.pretrained))
                print("=> Best loss: {} @ epoch {}".format(checkpoint["best_loss"], checkpoint["epoch"]))
        
        if args.finetune == "head":
            # Freeze all layers but the last fc
            for name, param in model.named_parameters():
                if 'fc_cls' not in name:
                    param.requires_grad = False
            logging.info("=> Finetune only head layer")
        
        model.head = head_model    
    elif args.model_name == "vivit":
        #model = vivit.ViViT(112, 8, 1, 75, pool='mean', scale_dim=8, heads=6, depth=8, dropout=0.3, emb_dropout=0.3).cuda()
        model = vivit.ViViT(112, 16, 1, 75, pool='mean', heads=2, dim_head=64, depth=2).cuda()
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

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)

    model = nn.DataParallel(model)
    model = model.to('cuda')
    #print(model)

    # Loss function
    #criterion = RMSLELoss().to(device)
    criterion = nn.L1Loss().to(device)
    #criterion = nn.SmoothL1Loss().to(device)
    #criterion = nn.MSELoss().to(device)

    # Optimise only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    
    optimiser = optim.SGD(parameters, lr=args.lr, weight_decay=0)
    #optimiser = optim.Adam(parameters, lr=args.lr, weight_decay=0.5)
    #optimiser = optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.5)

    #scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.5)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimiser,
                                        lr_lambda=lambda epoch: 0.99 ** epoch)
    # Load data
    augmentation = [RandomROI([0])]

    augmentation = Transformer(
            augmentation, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] #imagenet setting
        )

    if args.dataset_name == "mahnob":
        train_dataset = MAHNOBHCIDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = MAHNOBHCIDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        assert not [
            i for i in val_dataset.files if i in train_dataset.files
        ], "Train/Val datasets are intersected!"

    elif args.dataset_name == "vipl-hr-v2":

        train_dataset = VIPLHRDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = VIPLHRDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )

    elif args.dataset_name == "ubfc2":

        train_dataset = UBFC2Dataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = UBFC2Dataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
    elif args.dataset_name == "ubfc1":

        train_dataset = UBFC1Dataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = UBFC1Dataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
    elif args.dataset_name == "merged":
        
        train_dataset = MergedDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = MergedDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
    elif args.dataset_name == "pure":
        
        train_dataset = PUREDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = PUREDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
    elif args.dataset_name == "cohface":
        
        train_dataset = CohfaceDataset(
            args.dataset_dir,
            True,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )
        val_dataset = CohfaceDataset(
            args.dataset_dir,
            False,
            augmentation,
            args.vid_frame,
            args.vid_frame_stride,
        )

    else:
        print("Unsupported datasets!")
        return

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
            train_loss = train(train_loader, model, criterion, optimiser, device, args.finetune == "head")
            #scheduler.step()
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

def train(train_loader, model, criterion, optimizer, device, finetune = True):
    losses = AverageMeter("Loss", ":.4e")
    

    if finetune:
        model.eval() 
    else:
        model.train()
    #head_model.train()

    pbar = tqdm(train_loader, desc="Train Iteration")
    tasum = 0; pasum = 0
    cnt  = 0
    for videos, targets in pbar:
        # Process input
        videos = videos.to(device, non_blocking=True)
        targets = targets.reshape(-1, 1).to(device, non_blocking=True)
        # Compute output
        # for name, param in model.named_parameters():
        #     if 'patch' in name:
        #         print('debug 403 before update', name, param.data)
        preds = model(videos)
        # Loss
        loss = criterion(preds, targets)
        # std between preds and targets
        #stddiff = torch.std(preds) - torch.std(targets)
        #loss = loss * 0.9 + stddiff * 0.1 #ycyoon 값이 몰리지 않도록 조정
        losses.update(loss.item(), videos.size(0))

        # Compute gradient
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        te = targets[-20:].tolist(); pe = preds[-20:].tolist()
        ta = [round(t[0], 2) for t in te]; pa = [round(p[0], 2) for p in pe]
        tasum += round(np.std(ta),2); pasum += round(np.std(pa),2)
        cnt += 1
        pbar.set_description("loss %f std pred:%f" % (losses.avg, pasum/cnt))

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
    pasum = 0; tasum = 0
    #headmodel.eval()
    for videos, targets in tqdm(val_loader, desc="Val Iteration"):
        # for name, param in model.named_parameters():
        #     if 'patch' in name:
        #         print('debug 430 validate', name, param.data)
        # Process input
        videos = videos.to(device, non_blocking=True)
        targets = targets.reshape(-1, 1).to(device, non_blocking=True)

        # Compute output
        preds = model(videos)
        # Loss
        mae = criterion(preds, targets)
        mse = mse_loss_func(preds, targets)
        te = targets[-20:].tolist(); pe = preds[-20:].tolist()
        ta = [round(t[0], 2) for t in te]; pa = [round(p[0], 2) for p in pe]
        # std of ta
        
        #print('example', [list(a) for a in zip(ta, pa)], round(np.std(ta),2), round(np.std(pa),2))
        #logging.info('example', [list(a) for a in zip(ta, pa)], round(np.std(ta),2), round(np.std(pa),2))
        tasum += np.std(ta); pasum += np.std(pa)

        maes.update(mae.item(), targets.size(0))
        mses.update(mse.item(), targets.size(0))

        all_pred.append(preds.detach().cpu())
        all_true.append(targets.detach().cpu())
    print('std of targets', tasum/len(val_loader), 'std of preds', pasum/len(val_loader))

    all_pred = torch.cat(all_pred).flatten()
    all_true = torch.cat(all_true).flatten()

    # Mean and Std
    diff = all_pred - all_true
    mean = torch.mean(diff)
    std = torch.std(diff)

    # MSE
    mse_loss = mses.avg

    # RMSE
    rmse_loss = np.sqrt(mse_loss)
    
    r, _ = pearsonr(all_true, all_pred)

    return maes.avg, std, rmse_loss, r


if __name__ == "__main__":
    main()
