import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import torchvision
from dataset import DetectionDataset
from unet import UNet
from torch import optim
from torch.utils.data import DataLoader
from transform import *
from torchvision import transforms
import segmentation_models_pytorch as smp

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.insert(0, os.path.abspath((os.path.dirname(__file__)) + '/../'))
from utils import get_logger, dice_coeff, dice_loss



def eval_net(net, dataset, device):
    net.eval()
    tot = 0.
    with torch.no_grad():
        for i, b in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
            imgs, true_masks = b
            result = net(imgs.to(device))
            masks_pred = result[:, 0].squeeze(1)  # (b, 1, h, w) -> (b, h, w)
            masks_pred = (F.sigmoid(masks_pred) > 0.5).float()
            tot += dice_coeff(masks_pred.cpu(), true_masks).item()
    return tot / len(dataset)


def train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, logger, args=None, device=None):
    num_batches = len(train_dataloader)

    best_model_info = {'epoch': -1, 'val_dice': 0., 'train_dice': 0., 'train_loss': 0.}

    for epoch in range(args.epochs):
        logger.info('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        net.train()
        if scheduler is not None:
            scheduler.step(epoch)

        epoch_loss = 0.
        tqdm_iter = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        mean_bce, mean_dice = [], []
        for i, batch in tqdm_iter:
            imgs, true_masks = batch
            result = net(imgs.to(device))
            masks_pred = result[:, 0]
            masks_probs = F.sigmoid(masks_pred)

            bce_val, dice_val = criterion(masks_probs.cpu().view(-1), true_masks.view(-1))
            loss = bce_val + dice_val
            
            mean_bce.append(bce_val.item())
            mean_dice.append(dice_val.item())
            epoch_loss += loss.item()
            tqdm_iter.set_description('mean loss: {:.4f}'.format(epoch_loss / (i + 1)))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        logger.info('Epoch finished! Loss: {:.5f} ({:.5f} | {:.5f})'.format(epoch_loss / num_batches,
                                                                            np.mean(mean_bce), np.mean(mean_dice)))

        val_dice = eval_net(net, val_dataloader, device=device)
        if val_dice > best_model_info['val_dice']:
            best_model_info['val_dice'] = val_dice
            best_model_info['train_loss'] = epoch_loss / num_batches
            best_model_info['epoch'] = epoch
            torch.save(net.state_dict(), os.path.join(args.output_dir, f'{args.model}_CP-best_epoch-{epoch}.pth'))
            logger.info('Validation Dice Coeff: {:.5f} (best)'.format(val_dice))
        else:
            logger.info('Validation Dice Coeff: {:.5f} (best {:.5f})'.format(val_dice, best_model_info['val_dice']))

        torch.save(net.state_dict(), os.path.join(args.output_dir, f'{args.model}_best_epoch-{epoch}.pth'))
        
def main():
    parser = ArgumentParser()
    parser.add_argument('-d', '--data_path', dest='data_path', type=str, default=None ,help='path to the data')
    parser.add_argument('-e', '--epochs', dest='epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('-b', '--batch_size', dest='batch_size', default=40, type=int, help='batch size')
    parser.add_argument('-s', '--image_size', dest='image_size', default=256, type=int, help='input image size')
    parser.add_argument('-lr', '--learning_rate', dest='lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('-wd', '--weight_decay', dest='weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('-lrs', '--learning_rate_step', dest='lr_step', default=10, type=int, help='learning rate step')
    parser.add_argument('-lrg', '--learning_rate_gamma', dest='lr_gamma', default=0.5, type=float,
                        help='learning rate gamma')
    parser.add_argument('-m', '--model', dest='model', default='fpn',)
    parser.add_argument('-w', '--weight_bce', default=0.5, type=float, help='weight BCE loss')
    parser.add_argument('-l', '--load', dest='load', default=False, help='load file model')
    parser.add_argument('-v', '--val_split', dest='val_split', default=0.7, help='train/val split')
    parser.add_argument('-o', '--output_dir', dest='output_dir', default='./output', help='dir to save log and models')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info('Start training with params:')
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)
    
#     net = UNet() # TODO: to use move novel arch or/and more lightweight blocks (mobilenet) to enlarge the batch_size
#     net = smp.FPN('mobilenet_v2', encoder_weights='imagenet', classes=2)
    net = smp.FPN('se_resnet50', encoder_weights='imagenet', classes=2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.load:
        net.load_state_dict(torch.load(args.load))
    logger.info('Model type: {}'.format(net.__class__.__name__))

    net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = lambda x, y: (args.weight_bce * nn.BCELoss()(x, y), (1. - args.weight_bce) * dice_loss(x, y))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma) \
        if args.lr_step > 0 else None

    train_transforms = Compose([
        Crop(min_size=1 - 1 / 3., min_ratio=1.0, max_ratio=1.0, p=0.5),
        Flip(p=0.05),
        RandomRotate(),
        Pad(max_size=0.6, p=0.25),
        Resize(size=(args.image_size, args.image_size), keep_aspect=True),
        ScaleToZeroOne(),
    ])
    val_transforms = Compose([
        Resize(size=(args.image_size, args.image_size)),
        ScaleToZeroOne(),
    ])
    
    train_dataset = DetectionDataset(args.data_path, os.path.join(args.data_path, 'train_mask.json'),
                                     transforms=train_transforms)
    val_dataset = DetectionDataset(args.data_path, None, transforms=val_transforms)

    train_size = int(len(train_dataset) * args.val_split)
    val_dataset.image_names = train_dataset.image_names[train_size:]
    val_dataset.mask_names = train_dataset.mask_names[train_size:]
    train_dataset.image_names = train_dataset.image_names[:train_size]
    train_dataset.mask_names = train_dataset.mask_names[:train_size]
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                  shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4,
                                shuffle=False, drop_last=False)
    logger.info('Number of batches of train/val=%d/%d', len(train_dataloader), len(val_dataloader))
    
    try:
        train(net, optimizer, criterion, scheduler, train_dataloader, val_dataloader, logger=logger, args=args,
              device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), os.path.join(args.output_dir, f'{args.model}_INTERRUPTED.pth'))
        logger.info('Saved interrupt')
        sys.exit(0)
 

if __name__ == '__main__':
    main()
