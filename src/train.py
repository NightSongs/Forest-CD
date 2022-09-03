# -*- coding: utf-8 -*-
import glob
import random
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from apex import amp
from tqdm import tqdm

import change_detection_pytorch as cdp
from change_detection_pytorch.losses import DiceLoss, FocalLoss
from utils.dataset import get_dataloader
from utils.metrics import cal_val_f1
from utils.trick import random_scale, GradualWarmupScheduler, sample_cutmix

warnings.filterwarnings('ignore')
torch.backends.cudnn.enabled = True
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def check_tensor(data, is_label):
    if not is_label:
        return data if data.ndim <= 4 else data.squeeze()
    return data if data.ndim <= 3 else data.squeeze()


# 固定随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 加载模型
def load_model(DEVICE, encoder, channels, pretrain, pretrain_model_path):
    model = cdp.UPerNet(
        encoder_name="{}".format(encoder),
        encoder_weights="imagenet",
        in_channels=channels,
        classes=2,
        siam_encoder=True,
        fusion_form='concat',
        ocr_head=True
    )
    model.to(DEVICE)
    if pretrain:
        model.load_state_dict(torch.load(pretrain_model_path))
    return model


#  混合loss
class hyperloss(nn.Module):
    def __init__(self):
        super(hyperloss, self).__init__()
        self.DiceLoss_fn = DiceLoss(mode='multiclass')
        self.FocalLoss_fn = FocalLoss(mode='multiclass', gamma=2, alpha=0.25)

    def forward(self, pred, mask):
        loss_focal = self.FocalLoss_fn(pred, mask)
        loss_dice = self.DiceLoss_fn(pred, mask)
        loss = loss_focal + loss_dice
        return loss


# 定义优化器及损失函数
def load_opt(model, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2, eta_min=int(1e-5))
    loss_cd = hyperloss().cuda()
    return optimizer, scheduler, loss_cd


# 生成dataloader
def build_dataloader(train_path, val_path, batch_size):
    train_loader = get_dataloader(train_path[0], train_path[1], train_path[2],
                                  "train", batch_size, shuffle=True, num_workers=4, drop_last=True)
    valid_loader = get_dataloader(val_path[0], val_path[1], val_path[2],
                                  "val", batch_size, shuffle=False, num_workers=4, drop_last=True)
    return train_loader, valid_loader


# 训练函数
def train(num_epochs, optimizer, scheduler, loss_fn, train_loader, valid_loader, model, save_path, use_apex=False,
          warmUp=False, multi_scale=False):
    if warmUp:
        scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    epochs = num_epochs + 1
    header = r'Epoch/EpochNum | TrainLoss | Valid-F1 | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.3f} | {:9.3f} | {:9.2f}'
    print(header)
    # 记录当前验证集最优f1,以判定是否保存当前模型
    best_f1 = 0
    best_f1_epoch = 0
    train_loss_epochs, val_f1_epochs, lr_epochs = [], [], []
    for epoch in range(1, epochs):
        model.train()
        losses = []
        start_time = time.time()
        for batch_index, (imageA, imageB, target) in enumerate(tqdm(train_loader)):
            imageA, imageB, target = check_tensor(imageA, False), check_tensor(imageB, False), \
                                     check_tensor(target, True)
            imageA, imageB, target = sample_cutmix(imageA, imageB, target)
            if multi_scale:
                ms = random.choice([True, False])
                if ms:
                    scale = random.uniform(0.75, 1.25)
                    imageA, imageB, target = random_scale(imageA, imageB, target, imageA.shape[2:], (scale, scale))
            imageA, imageB, target = imageA.float(), imageB.float(), target.long()
            imageA, imageB, target = imageA.to(DEVICE), imageB.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(imageA, imageB)
            # 对监督对象区域估计（或辅助损失）的损失的权重为0.4[coarse_pre, pre]:
            loss = loss_fn(output[0], target) * 0.4 + loss_fn(output[1], target)
            if use_apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()
            losses.append(loss.item())
        if warmUp:
            scheduler.step(epoch)
        else:
            scheduler.step()
        val_iou = cal_val_f1(model, valid_loader)
        train_loss_epochs.append(np.array(losses).mean())
        val_f1_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        print(raw_line.format(epoch, num_epochs, np.array(losses).mean(),
                              np.mean(val_iou),
                              (time.time() - start_time) / 60 ** 1), end="")
        if best_f1 < np.stack(val_iou).mean(0).mean():
            best_f1 = np.stack(val_iou).mean(0).mean()
            best_f1_epoch = epoch
            torch.save(model, save_path)
            print("  valid f1 is improved. the model is saved.")
        else:
            print("")
            if (epoch - best_f1_epoch) >= 20:
                break
    return train_loss_epochs, val_f1_epochs, lr_epochs


if __name__ == '__main__':
    setup_seed(1021)
    num_epochs = 100
    batch_size = 16
    channels = 3
    lr = 1e-4
    use_apex = False
    warmUp = False
    multi_scale = True
    encoder_list = ["Swin-T"]
    train_dataset = [sorted(glob.glob("/train/A/*.tif")), sorted(glob.glob("/train/B/*.tif")),
                     sorted(glob.glob("/train/label/*.png"))]
    val_dataset = [sorted(glob.glob("/val/A/*.tif")), sorted(glob.glob("/val/B/*.tif")),
                   sorted(glob.glob("/val/label/*.png"))]
    for encoder in encoder_list:
        model = load_model(DEVICE, encoder, channels, pretrain=False,
                           pretrain_model_path=None)
        optimizer, scheduler, loss_fn = load_opt(model, lr)
        if use_apex:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        model_save_path = "../user_data/model_data/change_detection_{}_upernet.pth".format(encoder)
        train_loader, valid_loader = build_dataloader(train_dataset, val_dataset, int(batch_size))
        train_loss_epochs, val_f1_epochs, lr_epochs = train(num_epochs, optimizer, scheduler, loss_fn,
                                                            train_loader, valid_loader, model, model_save_path,
                                                            use_apex, warmUp, multi_scale)
