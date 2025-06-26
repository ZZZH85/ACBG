import datetime
import time
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

import joint_transforms
from config import  new_training_root
from config import backbone_path
from datasets_transform_diff import ImageFolder
from misc import AvgMeter, check_mkdir
from ACBG import ACBG

import CCAMlosshardsample
import numpy as np
import warnings

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
cudnn.benchmark = True

device_ids = [0]

ckpt_path = './ckpt'
exp_name = 'ACBG'
args = {
    'epoch_num': 30,
    'train_batch_size': 8,
    'last_epoch': 0,
    'lr': 1e-5,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 512,
    'poly_train': True,
    'optimizer': 'Adam',
}

print(torch.__version__)

# Path.
check_mkdir(ckpt_path)
check_mkdir(os.path.join(ckpt_path, exp_name))
vis_path = os.path.join(ckpt_path, exp_name, 'log')
check_mkdir(vis_path)
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')
writer = SummaryWriter(log_dir=vis_path, comment=exp_name)

# Transform Data.
joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])

img_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# Prepare Data Set.
train_set = ImageFolder(new_training_root)
print("Train set: {}".format(train_set.__len__()))
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=16, shuffle=True)

total_epoch = args['epoch_num'] * len(train_loader)

# loss function
structure_loss = CCAMlosshardsample.structure_loss().cuda(device_ids[0])
bce_loss = nn.BCEWithLogitsLoss().cuda(device_ids[0])
iou_loss = CCAMlosshardsample.IOU().cuda(device_ids[0])

bb_loss = CCAMlosshardsample.SimMaxLoss().cuda(device_ids[0])
ff_loss = CCAMlosshardsample.SimMaxLoss().cuda(device_ids[0])
bf_loss = CCAMlosshardsample.SimMinLoss().cuda(device_ids[0])  # 最小化相似性


def bce_iou_loss(pred, target):
    bce_out = bce_loss(pred, target)
    iou_out = iou_loss(pred, target)

    loss = bce_out + iou_out

    return loss


def cal_eroded(smp_gt):
    gt0 = smp_gt[:, 0, :, :].float().unsqueeze(1)  # 前景
    # gt1 = 1- smp_gt[:,0,:,:].float().unsqueeze(1)  #背景
    max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)  # 可调整 kernel_size 和 padding
    # 假设要处理的张量为 tensor
    # ===== 膨胀 =====
    tensor_dilate = max_pool(gt0)
    tensor_dilate = max_pool(tensor_dilate)
    tensor_dilate = max_pool(tensor_dilate)
    gt1 = abs(tensor_dilate - gt0)  # 先处理边缘
    return gt0, gt1


def cal_intra_loss(smp, fore_labels1, eroded_labels1):  # smp 特征图
    criterion = [CCAMlosshardsample.SimMaxLoss(metric='cos').cuda(), CCAMlosshardsample.SimMinLoss(metric='cos').cuda(),
                 CCAMlosshardsample.SimMaxLoss(metric='cos').cuda()]
    sum_cosine_dis = torch.zeros(16).cuda()
    bs = smp.shape[0]
    channels = smp.shape[1]
    for i in range(0, bs):
        batch_smp = smp[i]

        batch_fore_labels1 = fore_labels1[i]
        batch_fore_labels1_test = batch_fore_labels1.expand_as(batch_smp)
        idx0 = torch.where(batch_fore_labels1_test > 0.5)
        temp_fore = batch_smp[idx0]
        temp_fore = temp_fore.reshape(channels, -1)   #前景特征 64*point
        idx_fore = np.array(range(0, len(temp_fore[1])))
        select_fore_point = len(temp_fore[1])
        select_fore_point = min(500, select_fore_point)


        batch_eroded_labels1 = eroded_labels1[i]
        batch_eroded_labels1_test = batch_eroded_labels1.expand_as(batch_smp)
        idx1 = torch.where(batch_eroded_labels1_test > 0.5)
        temp_eroded = batch_smp[idx1]
        temp_eroded = temp_eroded.reshape(channels, -1)   #背景特征 64*point
        idx_eroded = np.array(range(0, len(temp_eroded[1])))
        select_eroded_point = len(temp_eroded[1])
        select_eroded_point = min(500, select_eroded_point)

        #
        np.random.shuffle(idx_fore)
        idx_fore = idx_fore[:select_fore_point]
        temp_fore_inv = temp_fore[:, idx_fore].permute(1,0)

        np.random.shuffle(idx_eroded)
        idx_eroded = idx_eroded[:select_eroded_point]
        temp_eroded_inv = temp_eroded[:, idx_eroded].permute(1,0)

        #背景背景
        batch_bb_loss=criterion[0](temp_eroded_inv)
        # batch_bb_loss_point =torch.max(batch_bb_loss,dim=0)  #取最大距离
        temp_batch_bb_loss_point = torch.sort(batch_bb_loss, dim=0)
        temp_select_eroded_point = int(select_eroded_point / 2)
        temp_batch_bb_loss_point = temp_batch_bb_loss_point[0]  #取出sort后的值
        temp_batch_bb_loss_point = temp_batch_bb_loss_point[temp_select_eroded_point:,:]
        batch_bb_loss_point =torch.mean(temp_batch_bb_loss_point)


        #前景前景
        batch_ff_loss = criterion[2](temp_fore_inv)
        # batch_ff_loss_point =torch.max(batch_ff_loss,dim=0)  #按行取最大值
        temp_batch_ff_loss_point=torch.sort(batch_ff_loss,dim=0)
        temp_select_fore_point=int(select_fore_point/2)
        temp_batch_ff_loss_point=temp_batch_ff_loss_point[0]
        temp_batch_ff_loss_point=temp_batch_ff_loss_point[temp_select_fore_point:,:]
        batch_ff_loss_point = torch.mean(temp_batch_ff_loss_point)


        #前景背景（背景）
        batch_bf_loss=criterion[1](temp_fore_inv,temp_eroded_inv)
        # batch_bf_loss_point = torch.max(batch_bf_loss, dim=0)
        # batch_bf_loss_point = torch.mean(batch_bf_loss_point[0])
        temp_batch_bf_loss_point=torch.sort(batch_bf_loss,dim=0)
        # temp_select_eroded_point = int(select_eroded_point / 2)
        temp_select_fore_point = int(select_fore_point / 2)  # 取和背景最近的前half个前景
        temp_batch_bf_loss_point=temp_batch_bf_loss_point[0]
        temp_batch_bf_loss_point = temp_batch_bf_loss_point[temp_select_fore_point:,:]
        # temp_batch_bf_loss_point=temp_batch_bf_loss_point[temp_select_eroded_point:,:]
        batch_bf_loss_point = torch.mean(temp_batch_bf_loss_point)



        batch_loss=batch_bb_loss_point+batch_ff_loss_point+batch_bf_loss_point
        if select_fore_point==0:
            batch_loss=torch.zeros_like(batch_loss)
        if select_eroded_point ==0:
            batch_loss=torch.zeros_like(batch_loss)
        sum_cosine_dis[i] = batch_loss
    sum_loss = torch.mean((sum_cosine_dis)).cuda()
    return sum_loss


def main():
    print(args)
    print(exp_name)

    net = ACBG(backbone_path).cuda(device_ids[0]).train()

    if args['optimizer'] == 'Adam':
        print("Adam")
        optimizer = optim.Adam([
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] == 'bias' and name[0:5] == 'layer'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] == 'bias' and name[0:5] != 'layer'],
             'lr': 20 * args['lr']},  # 非bakebone部分学习率 *10 开始
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] != 'bias' and name[0:5] == 'layer'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']},
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] != 'bias' and name[0:5] != 'layer'],
             'lr': 10 * args['lr'], 'weight_decay': args['weight_decay']}
        ])
    else:
        print("SGD")
        optimizer = optim.SGD([
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] == 'bias' and name[0:5] == 'layer'],
             'lr': 2 * args['lr']},
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] == 'bias' and name[0:5] != 'layer'],
             'lr': 20 * args['lr']},  # 非bakebone部分学习率 *10 开始
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] != 'bias' and name[0:5] == 'layer'],
             'lr': 1 * args['lr'], 'weight_decay': args['weight_decay']},
            {'params': [param for name, param in net.named_parameters() if
                        name[-4:] != 'bias' and name[0:5] != 'layer'],
             'lr': 10 * args['lr'], 'weight_decay': args['weight_decay']}
        ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('Training Resumes From \'%s\'' % args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        total_epoch = (args['epoch_num'] - int(args['snapshot'])) * len(train_loader)
        print(total_epoch)

    net = nn.DataParallel(net, device_ids=device_ids)
    print("Using {} GPU(s) to Train.".format(len(device_ids)))

    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer)
    writer.close()


def train(net, optimizer):
    curr_iter = 1
    start_time = time.time()

    for epoch in range(args['last_epoch'] + 1, args['last_epoch'] + 1 + args['epoch_num']):
        loss_record, loss_1_record, loss_2_record, loss_3_record, loss_4_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()
        loss_cl_record,loss_point_record = AvgMeter(), AvgMeter()
        train_iterator = tqdm(train_loader, total=len(train_loader))  # tqdm  可视化进度条
        for data in train_iterator:
            if args['poly_train']:
                base_lr = args['lr'] * (1 - float(curr_iter) / float(total_epoch)) ** args['lr_decay']
                optimizer.param_groups[0]['lr'] = 2 * base_lr
                optimizer.param_groups[1]['lr'] = 1 * base_lr

            inputs, labels, edge_gt,img_label = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda(device_ids[0])
            labels = Variable(labels).cuda(device_ids[0])
            edge_gt = Variable(edge_gt).cuda(device_ids[0])

            optimizer.zero_grad()

            predict_1, predict_2, predict_3, predict_4, predict3_edge, predict4_edge, focus1 ,focus2,focus3= net(inputs)

            # foucu1
            m1 = nn.MaxPool2d(kernel_size=4)
            m2 = nn.MaxPool2d(kernel_size=2)
            labels1 = m1(labels)
            fore_gt1, eroded_gt1 = cal_eroded(labels1)
            loss_cl1 = cal_intra_loss(focus1, fore_gt1, eroded_gt1)

            # foucus2
            # m2 = nn.MaxPool2d(kernel_size=8)
            labels2 = m2(labels)
            labels2 = m1(labels2)
            fore_gt2, eroded_gt2 = cal_eroded(labels2)
            loss_cl2 = cal_intra_loss(focus2, fore_gt2, eroded_gt2)

            # focus3
            # m3 = nn.MaxPool2d(kernel_size=16)
            labels3 = m1(labels)
            labels3 = m2(labels3)
            labels3 = m2(labels3)
            fore_gt3, eroded_gt3 = cal_eroded(labels3)
            loss_cl3 = cal_intra_loss(focus3, fore_gt3, eroded_gt3)

            loss_cl= loss_cl1 + loss_cl2 + loss_cl3
            #
            loss_1 = bce_iou_loss(predict_1, labels)
            loss_2 = structure_loss(predict_2, labels)
            loss_3 = structure_loss(predict_3, labels)
            loss_4 = structure_loss(predict_4, labels)

            loss_point = structure_loss(predict3_edge, edge_gt.float()) + structure_loss(predict4_edge, edge_gt.float())

            loss = 1 * loss_1 + 1 * loss_2 + 2 * loss_3 + 4 * loss_4 + loss_point + loss_cl
            loss.backward()

            optimizer.step()

            loss_record.update(loss.data, batch_size)
            loss_1_record.update(loss_1.data, batch_size)
            loss_2_record.update(loss_2.data, batch_size)
            loss_3_record.update(loss_3.data, batch_size)
            loss_4_record.update(loss_4.data, batch_size)
            loss_point_record.update(loss_point.data,batch_size)
            loss_cl_record.update(loss_cl.data,batch_size)

            if curr_iter % 10 == 0:
                writer.add_scalar('loss', loss, curr_iter)
                writer.add_scalar('loss_1', loss_1, curr_iter)
                writer.add_scalar('loss_2', loss_2, curr_iter)
                writer.add_scalar('loss_3', loss_3, curr_iter)
                writer.add_scalar('loss_4', loss_4, curr_iter)
                writer.add_scalar('loss_point', loss_point, curr_iter)
                writer.add_scalar('loss_spa', loss_cl, curr_iter)

            log = '[%3d], [%6d], [%.6f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % \
                      (epoch, curr_iter, base_lr, loss_record.avg, loss_1_record.avg, loss_2_record.avg,
                       loss_3_record.avg, loss_4_record.avg,loss_point_record.avg,loss_cl_record.avg)
            train_iterator.set_description(log)
            open(log_path, 'a').write(log + '\n')

            curr_iter += 1

        if epoch % 1 == 0:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            net.cuda(device_ids[0])

        if epoch >= args['epoch_num']:
            net.cpu()
            torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % epoch))
            print("Total Training Time: {}".format(str(datetime.timedelta(seconds=int(time.time() - start_time)))))
            print(exp_name)
            print("Optimization Have Done!")
            return


if __name__ == '__main__':
    main()