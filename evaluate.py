import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from numpy import mean

import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import cv2
from tqdm import tqdm
from sklearn import metrics
import math
from config import *
from misc import *
from ACBG import ACBG
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import warnings

warnings.filterwarnings("ignore")

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
freq_name = "freq"
exp_name = "BGM_CAF"
args = {
    # 'scale':512,
    'save_results': True
}

print(torch.__version__)

# 对一张图片先进行尺度变换，再进行转化为Tensor算子
img_transform = transforms.Compose([
    # transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('casia1', casia1_pth),
])

results = OrderedDict()


def read_annotations(data_path):
    lines = map(str.strip, open(data_path).readlines())
    data = []
    for line in lines:
        temp = line.split()
        if len(temp) == 1:
            sample_path = temp[0]
            mask_path = 'None'
            label = -1
        else:
            sample_path, mask_path, label = temp
            # CASIA1plus
            if 'TP' in sample_path:
                base, img = sample_path.split('/TP/')
                sample_path = base + '/Tp/' + img[3:]
            if 'mask' in mask_path:
                base, mask = mask_path.split('/mask/')
                mask_path = base + '/mask/' + mask[3:]


            sample_path = '/home/zhanghao/datasets' + sample_path[1:]
            mask_path = '/home/zhanghao/datasets' + mask_path[1:]

            label = int(int(label) > 0)
        data.append((sample_path, mask_path, label))
    return data


def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    # assert prediction.dtype == np.uint8
    # assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape
    gt1 = np.array(gt, dtype=np.float32)
    prediction1 = np.array(prediction, dtype=np.float32)

    y_gt = gt1.flatten()
    y_pred = prediction1.flatten()
    precision, recall, thresholds = metrics.precision_recall_curve(y_gt, y_pred)
    # PR曲线实则是以precision（精准率）和recall（召回率）这两个为变量而做出的曲线，其中recall为横坐标，precision为纵坐标。
    # 设定一系列阈值，计算每个阈值对应的recall和precision，即可计算出PR曲线各个点。
    auc_score = roc_auc_score(y_gt, y_pred)
    return precision, recall, auc_score


# 返回每个阈值下的precision, recall,auc_score  是一个

def cal_fmeasure(precision, recall):
    # max_fmeasure = ([(2 * p * r) / (p + r + 1e-10) for p, r in zip(precision, recall)])
    # return max_fmeasure
    fmeasure = [[(2 * p * r) / (p + r + 1e-10)] for p, r in zip(precision, recall)]
    fmeasure = np.array(fmeasure)
    fmeasure = fmeasure[fmeasure[:, 0].argsort()]

    max_fmeasure = fmeasure[-1, 0]
    return max_fmeasure


def calculate_pixel_f1(pd, gt):
    if np.max(pd) == np.max(gt) and np.max(pd) == 0:
        f1, iou = 1.0, 1.0
        return f1, 0.0, 0.0
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    f1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    return f1, precision, recall

def calculate_img_score(pd, gt):
    seg_inv, gt_inv = np.logical_not(pd), np.logical_not(gt)
    true_pos = float(np.logical_and(pd, gt).sum())
    false_pos = np.logical_and(pd, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, gt).sum()
    true_neg = float(np.logical_and(seg_inv, gt_inv).sum())
    acc = (true_pos + true_neg) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    sen = true_pos / (true_pos + false_neg + 1e-6)
    spe = true_neg / (true_neg + false_pos + 1e-6)
    f1 = 2 * sen * spe / (sen + spe)
    return acc, sen, spe, f1, true_pos, true_neg, false_pos, false_neg

f1s = []
f1ss = []
scores, labs = [], []
f1s = [[], [], []]

bestf1s = []
pixelaucs = []
results = []


y_true = []
y_pred = []
newscores=[]
avgscores=[]
avgscore=0
annotation_file='./data/CASIAv1plus.txt'
annotation = read_annotations(annotation_file)


if __name__ == '__main__':
    pth_path = './ckpt'  # 模型保存的文件夹
    path_list = sorted([pth_path + f for f in os.listdir(pth_path) if f.endswith('.pth')])
    print(path_list)
    for path in path_list:     #测试保存的每个模型的指标，输出auc和F1指标。
        epoch=int(path.split("/")[-1].split(".")[0])
        if  epoch  %1==0:
            net = ACBG(backbone_path).cuda(device_ids[0])  # 加载backbone
            net.load_state_dict(torch.load(path))  ####改！
            print('Load {} succeed!'.format(path))
            net.eval()
            f1s = []
            aucs = []
            f1s = [[], [], []]

            mask_array_lst, pred_array_lst, pixelpred_array_lst = [], [], []

            bestf1s = []
            f1ss = []
            scores, labs = [], []

            bestf1s = []
            pixelaucs = []
            macrof1s = []
            results = []

            with torch.no_grad():
                start = time.time()
                for name, root in to_test.items():  # 对字典 以列表返回可遍历的(键, 值) 元组数组
                    time_list = []
                    image_path = os.path.join(root, 'Tp')

                    if args['save_results']:
                        check_mkdir(os.path.join(results_path, exp_name, name))

                    maxsize = 1
                    for ix, (img_path, mask_path, lab) in enumerate(tqdm(annotation[800:])):
                        img = Image.open(img_path).convert('RGB')
                        w, h = img.size
                        img_name = os.path.basename(img_path)[:-4]

                        resizesizew = 512
                        resizesizeh = 512

                        resizeimg = transforms.Resize((resizesizeh, resizesizew))
                        img_resize = resizeimg(img)
                        img_var = Variable(img_transform(img_resize).unsqueeze(0)).cuda(device_ids[0])
                        img_show = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                        prediction1, prediction2, prediction3, prediction, edge_predict3, edge_predict4, focus1, focus2, focus3 = net(
                            img_var)
                        prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))


                        img_name = os.path.basename(img_path)[:-4]

                        try:
                            pixel_pred = prediction / 255.0
                            img_pred = pixel_pred
                        except:
                            print("not exists" )
                            continue
                        labs.append(lab)
                        fixf1 = 0
                        bestf1 = 0
                        F1 = 0
                        if lab != 0:  # 是篡改图像
                            try:
                                gt = cv2.imread(mask_path, 0) / 255.0
                            except:
                                pdb.set_trace()
                            if pixel_pred.shape != gt.shape:
                                print("size not match" )
                                continue
                            fix_pred = (pixel_pred > 0.5).astype(np.float64)

                            try:
                                gt = np.array(gt, dtype=int)
                                precision, recall, auc_score = cal_precision_recall_mae(pixel_pred, gt)
                                f1ss = cal_fmeasure(precision, recall)
                                bestf1 = np.max(np.array(f1ss))

                                fixf1, p, r = calculate_pixel_f1(fix_pred.flatten(), gt.flatten())

                            except:
                                import pdb

                                pdb.set_trace()


                            bestf1s.append(bestf1)
                            pixelaucs.append(auc_score)
                            f1s[lab - 1].append(fixf1)

                    pixel_auc = np.mean(pixelaucs)
                    bestf1 = np.mean(bestf1s)
                    macrof1 = np.mean(macrof1s)
                    meanf1 = np.mean(f1s[0] + f1s[1] + f1s[2])

                    print("pixel-fixf1: %.4f" % meanf1)
                    print("pixel_bestf1 is {:.3f}".format(bestf1))
                    print("pixel_auc is {:.3f}".format(pixel_auc))