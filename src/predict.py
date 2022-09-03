import glob
import os

import cv2
import numpy as np
import torch

from utils.dataset import get_dataloader

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def get_test_dataloader(test_dataset, batch_size):
    test_dataloader = get_dataloader(test_dataset[0], test_dataset[1], None, "test",
                                     batch_size, shuffle=False, num_workers=0, drop_last=False)
    return test_dataloader


def load_model(model_path):
    OCR = True
    model = torch.load(model_path)
    model.eval()
    model.to(DEVICE)
    if "(no_ocr)" in model_path:
        OCR = False
    return model, OCR


def predict(test_dataloader, model, OCR, output_path, model_name):
    with torch.no_grad():
        for x1, x2, path1 in test_dataloader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            output = model(x1, x2)
            if OCR:
                output = output[1]
                output = output.cpu().data.numpy()
            else:
                output = output.cpu().data.numpy()
            for i in range(output.shape[0]):
                predict = np.argmax(output[i], axis=0).astype(np.uint8) * 255
                img_name = os.path.split(path1[i])[1].split(".tif")[0]
                save_path = os.path.join(output_path, img_name + "_" + model_name + ".tif")
                cv2.imwrite(save_path, predict)


if __name__ == "__main__":
    test_dataset = [sorted(glob.glob("/test/A/*.tif")), sorted(glob.glob("/test/B/*.tif"))]
    test_batch_size = 8
    output_path = "../user_data/predict_result"
    dataloader = get_test_dataloader(test_dataset, test_batch_size)
    model_list = ["../user_data/model_data/change_detection_Swin-B_upernet.pth"]
    for model_path in model_list:
        model_name = os.path.split(model_path)[1].split(".pth")[0].split("change_detection_")[1]
        model, OCR = load_model(model_path)
        predict(dataloader, model, OCR, output_path, model_name)
