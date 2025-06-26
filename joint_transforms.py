
import random
import numpy as np
from PIL import Image
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        #assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)  # Image.FLIP_LEFT_RIGHT 水平翻转
        return img, mask

class RandomVerticalFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)  # Image.FLIP_LEFT_RIGHT 水平翻转
        return img, mask


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask):
        #assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Padding(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)  PIL: (w, h)

    def __call__(self, img, mask):
        #assert img.size == mask.size

        pad_width = max(0, self.size[0] - img.size[0])
        pad_height = max(0, self.size[1] - img.size[1])

        img = np.pad(np.array(img), ((0, pad_height), (0, pad_width), (0, 0)), mode='constant', constant_values=0)
        mask = np.pad(np.array(mask), ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

        return Image.fromarray(img), Image.fromarray(mask)

        # return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)

# class ZeroPad2d(object):
#     # Pads the input tensor boundaries with zero.
#     def __init__(self, padding):
#         super(ZeroPad2d, self).__init__(padding, 0)
#     def __call__(self, img, mask):
#         #assert img.size == mask.size
#         return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)