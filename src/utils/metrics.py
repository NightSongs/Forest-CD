# -*- coding: utf-8 -*-
"""
@Time    : 2022/2/25/025 16:15
@Author  : NDWX
@File    : dataProcess.py
@Software: PyCharm
"""
import numpy as np
import torch

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


#  验证集不需要梯度计算,加速和节省gpu空间
@torch.no_grad()
# 计算验证集f1
def cal_val_f1(model, loader):
    TP_sum, FN_sum, FP_sum = [], [], []
    model.eval()
    for image, target in loader:
        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)[1]
        output = output.argmax(1)
        TP, FN, FP = cal_f1(output, target)
        TP_sum.append(TP)
        FN_sum.append(FN)
        FP_sum.append(FP)

    p = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FP_sum) + 0.000001)
    r = np.sum(TP_sum) / (np.sum(TP_sum) + np.sum(FN_sum) + 0.000001)
    val_f1 = 2 * r * p / (r + p + 0.000001)
    return val_f1


# 计算F1-Score
def cal_f1(pred, mask):
    TP = ((pred == 1) & (mask == 1)).sum()
    FN = ((pred == 0) & (mask == 1)).sum()
    FP = ((pred == 1) & (mask == 0)).sum()
    return TP, FN, FP
