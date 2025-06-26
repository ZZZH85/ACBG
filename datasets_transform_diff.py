import os
import os.path
import random
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
# from albumentations import RandomCrop,Compose
from iml_transforms import get_albu_transforms,mask_albu_transforms,img_albu_transforms
from torchvision import transforms as T
from torchvision.transforms import functional
import torchvision.transforms as transform
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2

# TRAIN CASIA2.0
def make_dataset(root):
    # def make_dataset(root):
    image_path = os.path.join(root, 'Tp/')
    # mask_path = os.path.join(root, 'mask/')
    mask_path = os.path.join(root, 'Gt/')
    # img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('.jpg')]
    img_list = [image_path + f for f in os.listdir(image_path) if f.endswith('.jpg') or f.endswith('.tif') or f.endswith('.bmp')]
    # a = [(os.path.join(image_path, img_name), mask_path + img_name[40:-4] + '_gt.png') for img_name in img_list]

    a = []
    for img_name in img_list:
        name = os.path.basename(img_name)
        (filename, extension) = os.path.splitext(name)
        mask_name = mask_path + filename + '_gt.png'
        # mask_name = mask_path + filename + '.png'
        a.append((os.path.join(image_path, img_name), mask_name))

    return a

#
img_albu_transforms = albu.Compose([
    albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(transpose_mask=True),
])

mask_albu_transforms = albu.Compose([
    ToTensorV2()
])

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image =functional .crop(image, *crop_params)
        if target is not None:
            target = functional .crop(target, *crop_params)
        return image, target



class ImageFolder(data.Dataset):
    # image and gt should be in the same folder and have same filename except extended name (jpg and png respectively)
    def __init__(self, root):
        self.root = root
        self.imgs = make_dataset(root)
        self.train_transform = get_albu_transforms(type_='train')
        self.tp_transform = get_albu_transforms(type_='test')

        # self.img_trans = img_albu_transforms
        self.mask_trans =mask_albu_transforms

    def __getitem__(self, index):
        img_path, gt_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        img_oral = img
        target = Image.open(gt_path).convert('L')
        target_oral = target

        imgname = os.path.basename(img_path)

        #label
        if 'Au' in imgname:
            label = 0
            tp_img = np.array(img)  # H W C
            gt_img = np.array(target)  # H W C
            res_dict = self.train_transform(image=tp_img, mask=gt_img)

            tp_img = res_dict['image']  # [H,W,C](0,255)
            gt_img = res_dict['mask']  # [H,W](max=255)

            res_img = img_albu_transforms(image=tp_img)
            img = res_img["image"]

            res_gt = mask_albu_transforms(image=gt_img)
            target = res_gt["image"].to(torch.float32)
            target = target / 255.0
        if 'Tp' in imgname:
            label = 1
            tp_img = np.array(img)  # H W C
            gt_img = np.array(target)  # H W C
            res_dict = self.tp_transform(image=tp_img, mask=gt_img)

            tp_img = res_dict['image']  # [H,W,C](0,255)
            gt_img = res_dict['mask']  # [H,W](max=255)

            res_img = img_albu_transforms(image=tp_img)
            img = res_img["image"]

            res_gt = mask_albu_transforms(image=gt_img)
            target = res_gt["image"].to(torch.float32)
            target = target / 255.0

        max_pool_0 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # 可调整 kernel_size 和 padding

        # 原图算边缘
        # ===== 膨胀 =====
        tensor_dilate_0 = max_pool_0(target)
        # tensor_dilate_0 = max_pool_0(tensor_dilate_0)

        # ===== 腐蚀 =====
        tensor_erode = -max_pool(-target)
        # tensor_erode = -max_pool(-tensor_erode)

        edge = tensor_dilate_0-tensor_erode

        return img, target,edge ,label
        # return img, target,edge

    def __len__(self):
        return len(self.imgs)
