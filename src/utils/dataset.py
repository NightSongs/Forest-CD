# -*- coding: utf-8 -*-
import albumentations as A
import gdal
import torch.utils.data as D
from albumentations.pytorch import ToTensorV2


# 读取图像
def imgread(fileName):
    dataset = gdal.Open(fileName)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    data = dataset.ReadAsArray(0, 0, width, height)
    if (len(data.shape) == 3):
        data = data.transpose((1, 2, 0))
    return data


# 构建dataset
class change_dataset(D.Dataset):
    def __init__(self, image_A_paths, image_B_paths, label_paths, mode):
        self.image_A_paths = image_A_paths
        self.image_B_paths = image_B_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_A_paths)
        assert len(image_A_paths) == len(image_B_paths), '前后时相影像数量不匹配'
        self.test_transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})
        self.train_transform = A.Compose([
            #  空间变换
            A.OneOf([
                A.VerticalFlip(p=1),
                A.HorizontalFlip(p=1),
                A.Transpose(p=1),
            ], p=0.5),
            #  色彩变换
            A.OneOf([
                A.RandomGamma(p=1),
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
                A.MultiplicativeNoise(multiplier=(0.7, 1.3), p=1),
                A.GaussNoise(p=1)
            ], p=0.5),
            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})

    def __getitem__(self, index):
        imageA = imgread(self.image_A_paths[index])
        imageB = imgread(self.image_B_paths[index])
        if self.mode == "train":
            label = imgread(self.label_paths[index])
            transformed_data = self.train_transform(image=imageA, image_2=imageB, mask=label)
            imageA, imageB, label = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            return imageA, imageB, label
        elif self.mode == "val":
            label = imgread(self.label_paths[index])
            transformed_data = self.test_transform(image=imageA, image_2=imageB, mask=label)
            imageA, imageB, label = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']
            return imageA, imageB, label
        elif self.mode == "test":
            transformed_data = self.test_transform(image=imageA, image_2=imageB)
            imageA, imageB = transformed_data['image'], transformed_data['image_2']
            return imageA, imageB, self.image_A_paths[index]

    def __len__(self):
        return self.len


# 构建数据加载器
def get_dataloader(image_A_paths, image_B_paths, label_paths, mode, batch_size,
                   shuffle, num_workers, drop_last):
    dataset = change_dataset(image_A_paths, image_B_paths, label_paths, mode)
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True, drop_last=drop_last)
    return dataloader
