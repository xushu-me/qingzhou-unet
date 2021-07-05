import os
import cv2
import numpy as np
import torch
from numpy import core
import glob
import logging
from Timer import *
from _02PipeDatasetLoader import *
from _03Unet import *
from _04Loss import *


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda')
    torch.cuda.set_device(0)  # 选用GPU设备
    # 加载网络，图片单通道，分类为1。
    # net = UNet(n_channels=1, n_classes=1)
    net = UNet(in_channels=3, out_channels=1, init_features=4, WithActivateLast=True, ActivateFunLast=torch.sigmoid).to(device)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('../_04Training/Output/0700_dict.pt', map_location='cuda'))
    # 测试模式
    net.eval()
    # 读取所有图片路径
    tests_path = glob.glob('test/*.jpg')
    print(tests_path)
    # 遍历所有图片
    for test_path in tests_path:
        # 保存结果地址
        save_res_path =  test_path.split('.')[0] + '_res.jpg'
        print(save_res_path)
        # 读取图片
        img = Image.open(test_path)
        # plt.imshow(img)
        # plt.show()
        # 处理图片
        img_tensor = ValImgTransform(img)
        # 转为tensor
        # img_tensor = torch.from_numpy(img)
        plt.imshow(np.array(img_tensor).transpose(1, 2, 0))
        plt.show()
        # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
        img_tensor = img_tensor.to(device=device, dtype=torch.float32)
        img_tensor = img_tensor.reshape(1, 3, 128, 128)
        print(img_tensor.shape)
        # 预测
        pred = net(img_tensor)
        print(pred.shape)
        # 提取结果
        pred = np.array(pred.data.cpu()[0])[0]
        # 处理结果
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        print(pred.shape)
        pred = (Normalization(pred) * 255).astype(np.uint8)
        print(pred)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2RGB)
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        plt.imshow(pred)
        plt.show()
        # 保存图片
        cv2.imwrite(save_res_path, pred)
