import random

import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.transforms import transforms


class JointResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size
        else:
            raise RuntimeError("size参数请设置为int或者tuple")

    def __call__(self, bg, fg, mask):
        bg = bg.resize(self.size, Image.BILINEAR)
        fg = fg.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        # target = target.resize(self.size, Image.NEAREST)
        return bg, fg, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, bg, fg, mask):
        # assert bg.size == fg.size
        # assert fg.size == mask.size
        for t in self.transforms:
            bg, fg, mask = t(bg, fg, mask)
        return bg, fg, mask
