import torch
from torch.autograd import Variable
import os
import torch.nn as nn
import argparse
from datetime import datetime
from lib.pvt import CAFE
from utils.dataloader import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr, AvgMeter,adjust_lr_d
import torch.nn.functional as F
import numpy as np
import logging
import time
import random

def mean_dice_np(y_true, y_pred, **kwargs):
    """
    compute mean dice for binary segmentation map via numpy
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice
 
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask,reduction='mean')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()
#############################################################################################################


def test(model, path, dataset):

    data_path = os.path.join(path, dataset)
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    model.eval()
    num1 = len(os.listdir(gt_root))
    #print(f'总计：{num1}')
    test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
   
    for i in range(num1):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res, res1, res2, res3,res4 = model(image)
        # eval Dice
        res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        input = res
        target = np.array(gt)
        
        mean_dice=mean_dice_np(target,input)
        DSC=DSC+mean_dice

   
    return DSC / num1


def train(train_loader, model, optimizer, epoch, test_path):
    model.train()
    global best
    size_rates = [0.75, 1, 1.25]
    loss_record = AvgMeter()
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
            images, gts = pack
            
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images = F.interpolate(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.interpolate(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            P1, P2, P3, P4,P5= model(images)
            
            
            # ---- loss function ----
            loss_P1 = structure_loss(P1, gts)
            loss_P2 = structure_loss(P2, gts)
            loss_P3 = structure_loss(P3, gts)
            loss_P4 = structure_loss(P4, gts)
            loss_P5 = structure_loss(P5, gts)
            
           
            loss = loss_P1 + loss_P2 + loss_P3 + loss_P4+loss_P5
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            # ---- recording loss ----
            if rate == 1:
                loss_record.update(loss.data, opt.batchsize)
        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                  ' loss-all:[{:0.4f}],P1:[{:0.4f}] lr:[{:0.7f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(),loss_P1, optimizer.param_groups[0]['lr']))
            logging.info('{} Epoch[{:03d}/{:03d}], Step[{:04d}/{:04d}],'
                  ' loss-all:[{:0.4f}],P1:[{:0.4f}] lr:[{:0.7f}]'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_record.show(),loss_P1, optimizer.param_groups[0]['lr']))
    # save model
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    global dict_plot

    
    if (epoch + 1) % 1 == 0 and epoch>=50:
        meandice = test(model, test_path, 'test')
        dict_plot['test'].append(meandice)
        if meandice > best:
            print('#'*80)
            best = meandice
            #torch.save(model.state_dict(), save_path + 'PolypPVT.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best.pth')
            print(f'best meandice:{best}')
            print('#'*80)
            for dataset in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
                dataset_dice = test(model, test_path, dataset)
                
                print(dataset, ': ', dataset_dice)
                dict_plot[dataset].append(dataset_dice)



if __name__ == '__main__':

    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}

    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=150, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=True, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=16, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=352, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.5, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='../dataset/polyp/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='../dataset/polyp/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth/polyp/')
    parser.add_argument('--log_path', type=str,default='./log/')

    opt = parser.parse_args()

    if not os.path.exists(opt.log_path):
        os.makedirs(opt.log_path)
    logging.basicConfig(filename=f'{opt.log_path}train_log_{int(time.time())}.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    
    model = CAFE().cuda()
    

    best = 0

    params = model.parameters() #model.parameters()保存的是Weights和Bais参数的值。

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(params, opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    logging.info(optimizer)

    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
                              augmentation=opt.augmentation)
    total_step = len(train_loader)

    print("#" * 20, "Start Training", "#" * 20)

    for epoch in range(1, opt.epoch):
        if epoch in [50, 90,120]:
            adjust_lr_d(optimizer,opt.lr, epoch,opt.decay_rate)
        train(train_loader, model, optimizer, epoch, opt.test_path)

