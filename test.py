import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
from lib.pvt import CAFE
from utils.dataloader import test_dataset
import cv2
import matplotlib.pyplot as plt
from utils.cam_utils import GradCAM, show_cam_on_image
import torchvision.transforms as transforms
################################################################

def mean_iou_np(y_true, y_pred, **kwargs):
    """
    compute mean iou for binary segmentation map via numpy
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(y_pred * y_true), axis=axes) 
    mask_sum = np.sum(np.abs(y_true), axis=axes) + np.sum(np.abs(y_pred), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou

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

################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--pth_path', type=str, default='./model_pth/small_polyp/PolypPVT-best.pth')
    opt = parser.parse_args()
    model = CAFE()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
    print(f'powerpoint:{opt.pth_path}')
    for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
        dice_bank = []
        iou_bank = []
        data_path = '../dataset/small_polyp/TestDataset/{}'.format(_data_name)
        save_path = './result_map/small_polyp/{}/'.format(_data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = '{}/images/'.format(data_path)
        gt_root = '{}/masks/'.format(data_path)
        num1 = len(os.listdir(gt_root))
        test_loader = test_dataset(image_root, gt_root, 352)
        
        for i in range(num1):
            image, gt, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            save_gt=gt
            save_img=image
            image = image.cuda()
            P1,P2,P3,P4,P5 = model(image)
            res = F.upsample(P1, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            #######################################

            save_image_name = name

            cv2.imwrite(save_path+save_image_name, res*255)

            dice = mean_dice_np(gt, res)
            iou = mean_iou_np(gt, res)

            dice_bank.append(dice)
            iou_bank.append(iou)
        print('{}--Dice: {:.4f}, IoU: {:.4f}'. format(_data_name,np.mean(dice_bank), np.mean(iou_bank)))
