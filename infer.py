import time
import datetime

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict
from config import *
from misc import *

from ACBG import ACBG

torch.manual_seed(2021)
device_ids = [0]
torch.cuda.set_device(device_ids[0])

results_path = './results'
check_mkdir(results_path)
exp_name = 'best'
args = {
    'scale':512,
    'save_results': True
}

print(torch.__version__)

# 对一张图片先进行尺度变换，再进行转化为Tensor算子
img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

to_test = OrderedDict([
    ('casia1', casia1_pth),
])

results = OrderedDict()


y_true = []
y_pred = []



if __name__ == '__main__':
    pth_path = './ckpt'  # 模型保存的文件夹
    path_list = sorted([pth_path + f for f in os.listdir(pth_path) if f.endswith('.pth')])
    print(path_list)
    for path in path_list:
        net = ACBG(backbone_path).cuda(device_ids[0])  # 加载backbone
        net.load_state_dict(torch.load(path))  ####改！
        print('Load {} succeed!'.format(path))
        net.eval()
        with torch.no_grad():
            start = time.time()
            for name, root in to_test.items():  # 对字典 以列表返回可遍历的(键, 值) 元组数组
                time_list = []
                image_path = os.path.join(root, 'Tp')
                if args['save_results']:
                    check_mkdir(os.path.join(results_path, exp_name, name))

                img_list = [os.path.splitext(f)[0] for f in os.listdir(image_path) if f.endswith('jpg')]
                for idx, img_name in enumerate(img_list):
                    img = Image.open(os.path.join(image_path, img_name + '.jpg')).convert('RGB')

                    w, h = img.size
                    img_var = Variable(img_transform(img).unsqueeze(0)).cuda(device_ids[0])

                    start_each = time.time()
                    _, _, _,prediction, edg1, edg2 ,focus1,focus2,focus3=net(img_var)
                    time_each = time.time() - start_each
                    time_list.append(time_each)

                    prediction = np.array(transforms.Resize((h, w))(to_pil(prediction.data.squeeze(0).cpu())))
                    edg1 = np.array(transforms.Resize((h, w))(to_pil(edg1.data.squeeze(0).cpu())))
                    edg2 = np.array(transforms.Resize((h, w))(to_pil(edg2.data.squeeze(0).cpu())))

                    if args['save_results']:
                        Image.fromarray(prediction).convert('L').save(
                            os.path.join(results_path, exp_name, name, img_name + '.png'))

        end = time.time()
        print("Total Testing Time: {}".format(str(datetime.timedelta(seconds=int(end - start)))))