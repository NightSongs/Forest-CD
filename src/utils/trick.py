# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:24
@Author  : NDWX
@File    : trick.py
@Software: PyCharm
"""
import numpy as np
import torch
import torchvision.transforms.functional as vF
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler


#  warm up
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                    self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in
                         self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


#  多尺度训练
def random_scale(x1, x2, y, img_scale, ratio_range, div=32):
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.
        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.
        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.
        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = [int(img_scale[0] * ratio), int(img_scale[1] * ratio)]
        return scale

    scale = random_sample_ratio(img_scale, ratio_range)
    scale = [(s // div) * div for s in scale]
    x1 = vF.resize(x1, scale)
    x2 = vF.resize(x2, scale)
    y = vF.resize(y, scale)

    return x1, x2, y


#  快速傅里叶变换->风格统一
def style_transfer(source_image, target_image):
    h, w, c = source_image.shape
    out = []
    for i in range(c):
        source_image_f = np.fft.fft2(source_image[:, :, i])
        source_image_fshift = np.fft.fftshift(source_image_f)
        target_image_f = np.fft.fft2(target_image[:, :, i])
        target_image_fshift = np.fft.fftshift(target_image_f)

        change_length = 1
        source_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
        int(h / 2) - change_length:int(h / 2) + change_length] = \
            target_image_fshift[int(h / 2) - change_length:int(h / 2) + change_length,
            int(h / 2) - change_length:int(h / 2) + change_length]

        source_image_ifshift = np.fft.ifftshift(source_image_fshift)
        source_image_if = np.fft.ifft2(source_image_ifshift)
        source_image_if = np.abs(source_image_if)

        source_image_if[source_image_if > 255] = np.max(source_image[:, :, i])
        out.append(source_image_if)
    out = np.array(out)
    out = out.swapaxes(1, 0).swapaxes(1, 2)
    out = out.astype(np.uint8)
    return out


# 直方图匹配(后时相统一至前时相)
def histogram_matching(t2, t1, band_num, bit_num=8):
    def_t2 = t2.copy()
    bmax = 2 ** bit_num
    for b in range(band_num):
        hist1, _ = np.histogram(t2[:, :, b].ravel(), bmax, [0, bmax])
        hist2, _ = np.histogram(t1[:, :, b].ravel(), bmax, [0, bmax])
        # 获得累计直方图
        cdf1 = hist1.cumsum()
        cdf2 = hist2.cumsum()
        # 归一化处理
        cdf1_hist = hist1.cumsum() / cdf1.max()
        cdf2_hist = hist2.cumsum() / cdf2.max()
        # diff_cdf里是每2个灰度值比率间的差值
        diff_cdf = [[0 for i in range(bmax)] for j in range(bmax)]
        for i in range(bmax):
            for j in range(bmax):
                diff_cdf[i][j] = abs(cdf1_hist[i] - cdf2_hist[j])
        # 灰度级与目标灰度级的对应表
        lut = [0 for i in range(bmax)]
        for i in range(bmax):
            squ_min = diff_cdf[i][0]
            index = 0
            for j in range(bmax):
                if squ_min > diff_cdf[i][j]:
                    squ_min = diff_cdf[i][j]
                    index = j
            lut[i] = ([i, index])
        h = int(t1.shape[0])
        w = int(t1.shape[1])
        # 对原图像进行灰度值的映射
        for i in range(h):
            for j in range(w):
                def_t2[i, j, b] = lut[int(t2[i, j, b])][1]
    return def_t2


#  随机生成bbox
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


#  具有正样本采样的cutmix
def sample_cutmix(imageA, imageB, label):
    while True:
        lam = np.random.beta(1.0, 1.0)
        rand_index = np.random.randint(0, imageA.size()[0])
        bbx1, bby1, bbx2, bby2 = rand_bbox(imageA.size(), lam)
        bbox = label[rand_index][bbx1:bbx2, bby1:bby2]
        target = torch.sum(bbox == 1.0)
        num = torch.tensor(bbox.size()[0] * bbox.size()[1]).cuda()
        percent = torch.div(target, num)
        if torch.gt(percent, torch.tensor(0.25).cuda()):
            imageA[:, :, bbx1:bbx2, bby1:bby2] = imageA[rand_index, :, bbx1:bbx2, bby1:bby2]
            imageB[:, :, bbx1:bbx2, bby1:bby2] = imageB[rand_index, :, bbx1:bbx2, bby1:bby2]
            label[:, bbx1:bbx2, bby1:bby2] = label[rand_index, bbx1:bbx2, bby1:bby2]
            return imageA, imageB, label
