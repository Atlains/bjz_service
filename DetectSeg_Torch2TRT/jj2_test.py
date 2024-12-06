from __future__ import division

from models import *
from Yolov3_utils.utils import *
from Yolov3_utils.datasets import *

import os
import glob
import sys
import time
import datetime
import argparse

import torch.nn.functional as F
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from matplotlib.ticker import NullLocator

from torch2trt import torch2trt, TRTModule
from Unet_unet import UNet
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
from utils import head_MorAnalysis, vac_MorAnalysis, bbox_iou, \
    get_effective_residue, SingleImageConclusion, get_neck_data, cal_angle, get_cnt

# STMClient
# from stm_worker3 import STMWorker, BufferWorker, BatchWorker
# import torch.multiprocessing as mp
# import json
# from tqdm import tqdm
# from datetime import datetime
# import h5py
# import redis

STATUS_CLOSED = 'closed'
STATUS_INIT = 'initializing'
STATUS_CONNECTED = 'connected'
STATUS_NETERROR = 'net_error'
STATUS_ERROR = 'error'


def Write_Text(file_name, contant):
    # file_name = 'test.txt'
    with open(file_name, "w") as f:
        f.writelines(contant)


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class Detect():
    def __init__(self):
        super(Detect, self).__init__()

        # create model
        num_classes = 41
        resnet = models.__dict__[opt.arch]()
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        fc_features = resnet.fc.in_features
        resnet.fc = nn.Linear(fc_features, num_classes)

        if TensorRT is True:
            self.yolo_head = YOLOHead(config_path=opt.model_def)#.cuda().eval()

            self.yolo_model_trt = TRTModule()
            self.unet_model_trt = TRTModule()
            self.resnet_model_trt = TRTModule()
            self.unet_neck_model_trt = TRTModule()
            # DarknetBackbone convert to TensorRT
            bs_suffix = f"_bs{opt.batch_size}"
            if Half is True:
                self.yolo_model_trt.load_state_dict(torch.load(opt.yolo_weights_path.split('.')[0] + f'_Half_trt{bs_suffix}.pth'))
                # unet_model_trt.load_state_dict(torch.load(opt.unet_weights_path.split('.')[0] + '_Half_trt.pth'))
                self.unet_model_trt.load_state_dict(torch.load(opt.unet_weights_path.split('.')[0] + f'_trt{bs_suffix}.pth'))
                self.unet_neck_model_trt.load_state_dict(
                    torch.load(opt.unet_neck_weights_path.split('.')[0] + f'_trt{bs_suffix}.pth'))
                self.resnet_model_trt.load_state_dict(
                    torch.load(opt.resnet_weights_path.split('.')[0] + f'_Half_trt{bs_suffix}.pth'))
            else:
                self.yolo_model_trt.load_state_dict(torch.load(opt.yolo_weights_path.split('.')[0] + f'_trt{bs_suffix}.pth'))
                self.unet_model_trt.load_state_dict(torch.load(opt.unet_weights_path.split('.')[0] + f'_trt{bs_suffix}.pth'))
                self.unet_neck_model_trt.load_state_dict(
                    torch.load(opt.unet_neck_weights_path.split('.')[0] + f'_trt{bs_suffix}.pth'))
                self.resnet_model_trt.load_state_dict(torch.load(opt.resnet_weights_path.split('.')[0] + f'_trt{bs_suffix}.pth'))

            # self.yolo_model_trt.eval()
            # self.unet_model_trt.eval()
            # self.resnet_model_trt.eval()
            # self.unet_neck_model_trt.eval()
        else:
            if Half is True:
                self.yolo = Darknet(opt.model_def, img_size=opt.yolo_img_size).to(device).half()
                # unet_HeadVac = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=True).to(device).half()
                self.unet_HeadVac = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device)
                self.unet_neck = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device)
                self.resnet = resnet.to(device).half()
            else:
                self.yolo = Darknet(opt.model_def, img_size=opt.yolo_img_size).to(device)
                self.unet_HeadVac = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device)
                self.unet_neck = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device)
                self.resnet = resnet.to(device)

            # Load checkpoint weights
            self.yolo.load_state_dict(torch.load(opt.yolo_weights_path, map_location=device))
            self.unet_HeadVac.load_state_dict(torch.load(opt.unet_weights_path, map_location=device))
            self.unet_neck.load_state_dict(torch.load(opt.unet_neck_weights_path, map_location=device))
            self.resnet.load_state_dict(torch.load(opt.resnet_weights_path, map_location=device))

            # Set in evaluation mode: when forward pass, BatchNormalization and Dropout will be ignored
            self.yolo.eval()
            self.unet_HeadVac.eval()
            self.unet_neck.eval()
            self.resnet.eval()

            # dataloader = DataLoader(
            #     ImageFolder_SC(opt.image_folder, img_size=opt.yolo_img_size),
            #     batch_size=opt.batch_size,
            #     shuffle=False,
            #     num_workers=opt.n_cpu,
            # )

        self.classes = load_classes(opt.class_path)  # Extracts class labels from file

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.first_run()
        self.img_size = opt.yolo_img_size
        self.crop_size = 128
        self.orig_size = 400
        if opt.Debug or opt.ReturnImg:
            self.res_ims = []
            self.imgs_sharp = []
        self.img_id = 0

    def reset(self):
        if opt.Debug or opt.ReturnImg:
            self.res_ims = []
            self.imgs_sharp = []
        self.img_id = 0

    def first_run(self):
        # 第一次运行速度慢，先运行10次
        batch_size = opt.batch_size
        for i in range(10):
            if TensorRT:
                # yolo
                if Half:
                    test_data = torch.rand(size=(batch_size, 1, opt.yolo_img_size, opt.yolo_img_size)).cuda().half()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.yolo_img_size, opt.yolo_img_size)).cuda()
                print(test_data.dtype)
                outputs = self.yolo_model_trt(test_data)
                detections = self.yolo_head(outputs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)

                # unet
                if Half:
                    # test_data = torch.rand(size=(1, 1, opt.unet_img_size, opt.unet_img_size)).cuda().half()
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                head_pred, vac_pred = self.unet_model_trt(test_data)

                # unet_neck
                if Half:
                    # test_data = torch.rand(size=(1, 1, opt.unet_img_size, opt.unet_img_size)).cuda().half()
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                head_pred, vac_pred = self.unet_neck_model_trt(test_data)

                # resnet
                if Half:
                    test_data = torch.rand(size=(batch_size, 1, opt.resnet_img_size, opt.resnet_img_size)).cuda().half()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.resnet_img_size, opt.resnet_img_size)).cuda()
                output = self.resnet_model_trt(test_data)
                output = torch.softmax(output, dim=1)
            else:
                # yolo
                if Half:
                    test_data = torch.rand(size=(batch_size, 1, opt.yolo_img_size, opt.yolo_img_size)).cuda().half()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.yolo_img_size, opt.yolo_img_size)).cuda()
                detections = self.yolo(test_data)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)

                # unet
                if Half:
                    # test_data = torch.rand(size=(1, 1, opt.unet_img_size, opt.unet_img_size)).cuda().half()
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                head_pred, vac_pred = self.unet_HeadVac(test_data)

                # unet_neck
                if Half:
                    # test_data = torch.rand(size=(1, 1, opt.unet_img_size, opt.unet_img_size)).cuda().half()
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda()
                head_pred, vac_pred = self.unet_neck(test_data)

                # resnet
                if Half:
                    test_data = torch.rand(size=(batch_size, 1, opt.resnet_img_size, opt.resnet_img_size)).cuda().half()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.resnet_img_size, opt.resnet_img_size)).cuda()
                output = self.resnet(test_data)
                output = torch.softmax(output, dim=1)

    def pre_process(self, image):
        bs = image.shape[0]
        for i in range(bs):
            img_nd = torch.from_numpy(image[i, :, :]).float()
            mean1 = torch.mean(img_nd)
            std1 = torch.std(img_nd)
            img_trans = (img_nd - mean1 + 3.0 * std1) / 6 / std1
            img_trans = torch.clamp(img_trans, 0, 1)
            img = img_trans.unsqueeze(0)
            # Pad to square resolution
            # img, _ = pad_to_square(img, 0)
            # Resize
            input_img = resize(img, self.img_size)
            orig_img = np.expand_dims(image[i, :, :], axis=0)
            orig_img = np.concatenate([orig_img, orig_img, orig_img], axis=0)
            orig_img = torch.from_numpy(orig_img)
            if i == 0:
                input_imgs = input_img.unsqueeze(0)
                orig_imgs = orig_img.unsqueeze(0)
            else:
                input_imgs = torch.cat([input_imgs, input_img.unsqueeze(0)], dim=0)
                orig_imgs = torch.cat([orig_imgs, orig_img.unsqueeze(0)], dim=0)
        return input_imgs, orig_imgs

    def predict_FM(self, input_imgs, orig_imgs):
        # color = [random.randint(0, 255) for _ in range(3)]
        color = [181, 142, 13]
        # print(color)
        file_names = []
        res_crop_coordinates = []
        res_FM = []
        res_focus_pos = []

        img_paths = [str(i) for i in range(input_imgs.shape[0])]
        if Half:
            input_imgs = Variable(input_imgs.type(self.Tensor)).half()
            orig_imgs = Variable(orig_imgs.type(self.Tensor)).half()
        else:
            input_imgs = Variable(input_imgs.type(self.Tensor))
            orig_imgs = Variable(orig_imgs.type(self.Tensor))

        # Get detections
        with torch.no_grad():

            # annotations
            # YOLOv3  return tensor size [batch_size, 10647, 85]
            # YOLOv3-tiny return tensor size [batch_size, 2535, 85]
            # 10647 = 3×13×13 + 3×26×26 + 3×52×52
            # 2535 = 3×13×13 + 3×26×26
            # 85: 4 for coordinates, the 5th is confidence of bbox, latter 80 dims are coco80 probability

            # TensorRT speedup
            if TensorRT:
                detections = self.yolo_head(self.yolo_model_trt(input_imgs))
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)
            else:
                detections = self.yolo(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)

            for img_i, (path, detection) in enumerate(zip(img_paths, detections)):
                # filename = path.split("/")[-1].split(".")[0]
                self.img_id += 1
                filename = path
                file_names.append(filename)
                if opt.Debug or opt.ReturnImg:
                    im0 = orig_imgs[img_i, :, :, :].float().permute(1, 2, 0).cpu()
                    im0 = np.array(im0)

                if detection is not None:
                    detection = rescale_boxes(detection, opt.yolo_img_size, tuple(orig_imgs.shape[-2:]))
                    crop_coordinate = []
                    confs = []
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        # if int(cls_pred) == 0:
                        xyxy = [x1, y1, x2, y2]
                        if opt.Debug or opt.ReturnImg:
                            # label = '%s %.2f' % (classes[int(cls)], conf)
                            label = '%s %.2f' % (self.classes[0], conf)
                            plot_one_box(xyxy, im0, label=label, color=color)
                        if conf > 0.8:
                            crop_coordinate.append(xyxy)
                            confs.append(conf)

                    # 判断距中心最近的精子
                    crop_coordinate = torch.tensor(crop_coordinate)
                    confs = torch.tensor(confs)
                    if len(crop_coordinate) > 1:
                        crop_coordinate = crop_coordinate[confs.argmax(), :]
                        # cen_dist = torch.abs((crop_coordinate[:, 0] + crop_coordinate[:, 2]) / 2 - 200) + \
                        #            torch.abs((crop_coordinate[:, 1] + crop_coordinate[:, 3]) / 2 - 200)
                        # crop_coordinate = crop_coordinate[cen_dist.argmin(), :]
                    elif len(crop_coordinate) == 1:
                        crop_coordinate = crop_coordinate[0]
                    else:
                        crop_coordinate = None

                    res_crop_coordinates.append(crop_coordinate)

                    if crop_coordinate is not None:
                        cent_x = torch.clamp(((crop_coordinate[0] + crop_coordinate[2]) / 2).int(), self.crop_size // 2,
                                             orig_imgs.shape[-1] - self.crop_size // 2)
                        cent_y = torch.clamp(((crop_coordinate[1] + crop_coordinate[3]) / 2).int(), self.crop_size // 2,
                                             orig_imgs.shape[-1] - self.crop_size // 2)

                        # 获取感兴趣区域
                        # input_imgs: [bs, 1, 416, 416]
                        # orig_imgs:  [bs, 3, 400, 400]
                        orig_img = orig_imgs[img_i, 0, :, :]
                        Roi_orig = orig_img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                                   int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2)]
                        Roi_mean = torch.mean(Roi_orig)
                        Roi_std = torch.std(Roi_orig)
                        Roi_seg = (Roi_orig - Roi_mean + 3 * Roi_std) / 6 / Roi_std
                        Roi_seg = torch.clamp(Roi_seg, 0, 1)
                        Roi_seg = Roi_seg.unsqueeze(0).unsqueeze(0)
                        Roi_sharp = (Roi_seg - 0.5) / 0.5
                        Roi_orig = Roi_orig.unsqueeze(0).unsqueeze(0)
                        # print(max(Roi_seg), max(Roi_sharp))
                        # FM
                        FM = torch.std(Roi_orig.float())
                    else:
                        if Half:
                            Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor).half()
                        else:
                            Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor)
                        Roi_sharp = Roi_seg
                        Roi_orig = Roi_seg
                        FM = 1e6

                    if img_i == 0:
                        Roi_sharps = Roi_sharp
                        Roi_origs = Roi_orig
                        res_FM.append(FM)
                        if opt.Debug or opt.ReturnImg:
                            im0 = np.expand_dims(im0, axis=0)
                            ims = im0
                    else:
                        Roi_sharps = torch.cat([Roi_sharps, Roi_sharp], dim=0)
                        Roi_origs = torch.cat([Roi_origs, Roi_orig], dim=0)
                        res_FM.append(FM)
                        if opt.Debug or opt.ReturnImg:
                            im0 = np.expand_dims(im0, axis=0)
                            ims = np.concatenate([ims, im0], axis=0)
                else:
                    if Half:
                        Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor).half()
                    else:
                        Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor)
                    Roi_sharp = Roi_seg
                    Roi_orig = Roi_seg
                    FM = 1e6
                    if img_i == 0:
                        Roi_sharps = Roi_sharp
                        Roi_origs = Roi_orig
                        res_FM.append(FM)
                        res_crop_coordinates.append(None)
                        if opt.Debug or opt.ReturnImg:
                            im0 = np.expand_dims(im0, axis=0)
                            ims = im0
                    else:
                        Roi_sharps = torch.cat([Roi_sharps, Roi_sharp], dim=0)
                        Roi_origs = torch.cat([Roi_origs, Roi_orig], dim=0)
                        res_FM.append(FM)
                        res_crop_coordinates.append(None)
                        if opt.Debug or opt.ReturnImg:
                            im0 = np.expand_dims(im0, axis=0)
                            ims = np.concatenate([ims, im0], axis=0)

            # Sharp
            if TensorRT:
                output = self.resnet_model_trt(Roi_sharps)
                output = torch.softmax(output, dim=1)
                cls = torch.argmax(output, dim=1)
                focus_pos = (cls - 20) * 0.5
            else:
                output = self.resnet(Roi_sharps)
                output = torch.softmax(output, dim=1)
                cls = torch.argmax(output, dim=1)
                focus_pos = (cls - 20) * 0.5

            res_focus_pos.extend(focus_pos)
            if opt.Debug or opt.ReturnImg:
                self.res_ims.extend(ims)

        return res_focus_pos

    def predict(self, input_imgs, orig_imgs):
        # color = [random.randint(0, 255) for _ in range(3)]
        colors = [(181, 142, 13), (255, 0, 0), (0, 0, 255)]
        # print(color)
        file_names = []
        res_head_coordinates = []
        res_neck_coordinates = []
        res_FM = []
        res_sperm_num = []
        img_paths = [str(i) for i in range(input_imgs.shape[0])]
        if Half:
            input_imgs = Variable(input_imgs.type(self.Tensor)).half()
            orig_imgs = Variable(orig_imgs.type(self.Tensor)).half()
        else:
            input_imgs = Variable(input_imgs.type(self.Tensor))
            orig_imgs = Variable(orig_imgs.type(self.Tensor))

        # Get detections
        with torch.no_grad():

            # annotations
            # YOLOv3  return tensor size [batch_size, 10647, 85]
            # YOLOv3-tiny return tensor size [batch_size, 2535, 85]
            # 10647 = 3×13×13 + 3×26×26 + 3×52×52
            # 2535 = 3×13×13 + 3×26×26
            # 85: 4 for coordinates, the 5th is confidence of bbox, latter 80 dims are coco80 probability

            # TensorRT speedup
            if cal_detect_time:
                tic_infer = time.time()
            if TensorRT:
                detections = self.yolo_head(self.yolo_model_trt(input_imgs), opt.Half)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)
            else:
                detections = self.yolo(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)
            if cal_detect_time:
                print(f"Inference time of {opt.batch_size} images:{time.time()-tic_infer}")
                tic_post = time.time()

            for img_i, (path, detection) in enumerate(zip(img_paths, detections)):
                # filename = path.split("/")[-1].split(".")[0]
                filename = path
                file_names.append(filename)
                if opt.Debug or opt.ReturnImg:
                    im0 = orig_imgs[img_i, :, :, :].float().permute(1, 2, 0).cpu()
                    im0 = np.array(im0)

                if detection is not None:
                    detection = rescale_boxes(detection, opt.yolo_img_size, tuple(orig_imgs.shape[-2:]))
                    sperm_num = len(detection[detection[:, -1] == 0])
                    head_coordinate = []
                    neck_coordinate = []
                    confs_head = []
                    confs_neck = []
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                        if int(cls_pred) == 2:
                            break

                        xyxy = [x1, y1, x2, y2]
                        if conf > 0.8 and int(cls_pred) == 0:
                            head_coordinate.append(xyxy)
                            confs_head.append(conf)

                        if int(cls_pred) == 1:
                            neck_coordinate.append(xyxy)
                            confs_neck.append(conf)

                    # 判断距中心最近的精子
                    head_coordinate = torch.tensor(head_coordinate)
                    confs_head = torch.tensor(confs_head)
                    if len(head_coordinate) > 1:
                        head_coordinate = head_coordinate[confs_head.argmax(), :]
                    elif len(head_coordinate) == 1:
                        head_coordinate = head_coordinate[0]
                    else:
                        head_coordinate = None
                    # if opt.Debug or opt.ReturnImg:
                        # if head_coordinate is not None:
                        #     label = '%s %.2f' % (self.classes[int(0)], conf)
                        #     plot_one_box(head_coordinate, im0, label=label, color=colors[int(0)])

                    neck_coordinate = torch.tensor(neck_coordinate)
                    confs_neck = torch.tensor(confs_neck)
                    if head_coordinate is not None:
                        if len(neck_coordinate) > 1:
                            ious = []
                            for item in neck_coordinate:
                                ious.append(bbox_iou(head_coordinate, item))
                            ious = torch.tensor(ious)
                            if ious.max() > 0:
                                neck_coordinate = neck_coordinate[ious.argmax(), :]
                            else:
                                neck_coordinate = None
                        elif len(neck_coordinate) == 1:
                            iou = bbox_iou(head_coordinate, neck_coordinate[0])
                            if iou > 0:
                                neck_coordinate = neck_coordinate[0]
                            else:
                                neck_coordinate = None
                        else:
                            neck_coordinate = None
                    else:
                        neck_coordinate = None

                    # if opt.Debug or opt.ReturnImg:
                    #     if neck_coordinate is not None:
                    #         label = '%s %.2f' % (self.classes[int(1)], conf)
                    #         plot_one_box(neck_coordinate, im0, label=label, color=colors[int(1)])

                    if head_coordinate is not None:
                        cent_x = torch.clamp(((head_coordinate[0] + head_coordinate[2]) / 2).int(), self.crop_size // 2,
                                             orig_imgs.shape[-1] - self.crop_size // 2)
                        cent_y = torch.clamp(((head_coordinate[1] + head_coordinate[3]) / 2).int(), self.crop_size // 2,
                                             orig_imgs.shape[-1] - self.crop_size // 2)

                        res_head_coordinates.append([
                            (cent_x - self.crop_size // 2).item(),
                            (cent_y - self.crop_size // 2).item(),
                            (cent_x + self.crop_size // 2).item(),
                            (cent_y + self.crop_size // 2).item(),
                        ])

                        # 获取感兴趣区域
                        # input_imgs: [bs, 1, 416, 416]
                        # orig_imgs:  [bs, 3, 400, 400]
                        orig_img = orig_imgs[img_i, 0, :, :]
                        Roi_orig = orig_img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                                   int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2)]
                        Roi_mean = torch.mean(Roi_orig)
                        Roi_std = torch.std(Roi_orig)
                        Roi_seg = (Roi_orig - Roi_mean + 3 * Roi_std) / 6 / Roi_std
                        Roi_seg = torch.clamp(Roi_seg, 0, 1)
                        Roi_seg = Roi_seg.unsqueeze(0).unsqueeze(0)
                        Roi_sharp = (Roi_seg - 0.5) / 0.5
                        Roi_orig = Roi_orig.unsqueeze(0).unsqueeze(0)
                        # print(max(Roi_seg), max(Roi_sharp))
                        # FM
                        FM = torch.std(Roi_orig.float()).cpu().numpy()
                    else:
                        res_head_coordinates.append(None)
                        if Half:
                            Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor).half()
                        else:
                            Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor)
                        Roi_sharp = Roi_seg
                        Roi_orig = Roi_seg
                        FM = 1e6

                    if neck_coordinate is not None:
                        cent_x = torch.clamp(((neck_coordinate[0] + neck_coordinate[2]) / 2).int(), self.crop_size // 2,
                                             orig_imgs.shape[-1] - self.crop_size // 2)
                        cent_y = torch.clamp(((neck_coordinate[1] + neck_coordinate[3]) / 2).int(), self.crop_size // 2,
                                             orig_imgs.shape[-1] - self.crop_size // 2)

                        res_neck_coordinates.append([
                            (cent_x - self.crop_size // 2).item(),
                            (cent_y - self.crop_size // 2).item(),
                            (cent_x + self.crop_size // 2).item(),
                            (cent_y + self.crop_size // 2).item(),
                        ])

                        # 获取感兴趣区域
                        # input_imgs: [bs, 1, 416, 416]
                        # orig_imgs:  [bs, 3, 400, 400]
                        orig_img = orig_imgs[img_i, 0, :, :]
                        Roi_crop = orig_img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                                   int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2)]
                        Roi_mean = torch.mean(Roi_crop)
                        Roi_std = torch.std(Roi_crop)
                        Roi_seg_neck = (Roi_crop - Roi_mean + 3 * Roi_std) / 6 / Roi_std
                        Roi_seg_neck = torch.clamp(Roi_seg_neck, 0, 1)
                        Roi_seg_neck = Roi_seg_neck.unsqueeze(0).unsqueeze(0)
                    else:
                        res_neck_coordinates.append(None)
                        if Half:
                            Roi_seg_neck = torch.rand((1, 1, 128, 128)).type(self.Tensor).half()
                        else:
                            Roi_seg_neck = torch.rand((1, 1, 128, 128)).type(self.Tensor)

                    if self.img_id == 0:
                        self.Roi_segs = [Roi_seg]
                        self.Roi_segs_neck = [Roi_seg_neck]
                        self.Roi_sharps = [Roi_sharp]
                        self.Roi_origs = [Roi_orig]
                        if opt.Debug or opt.ReturnImg:
                            self.ims = [im0]
                    else:
                        self.Roi_segs.append(Roi_seg)
                        self.Roi_segs_neck.append(Roi_seg_neck)
                        self.Roi_sharps.append(Roi_sharp)
                        self.Roi_origs.append(Roi_orig)
                        if opt.Debug or opt.ReturnImg:
                            self.ims.append(im0)

                    if img_i == 0:
                        res_FM.append(FM)
                    else:
                        res_FM.append(FM)
                else:
                    sperm_num = 0
                    if Half:
                        Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor).half()
                    else:
                        Roi_seg = torch.rand((1, 1, 128, 128)).type(self.Tensor)
                    if Half:
                        Roi_seg_neck = torch.rand((1, 1, 128, 128)).type(self.Tensor).half()
                    else:
                        Roi_seg_neck = torch.rand((1, 1, 128, 128)).type(self.Tensor)

                    Roi_sharp = Roi_seg
                    Roi_orig = Roi_seg
                    FM = 1e6
                    if self.img_id == 0:
                        self.Roi_segs = [Roi_seg]
                        self.Roi_segs_neck = [Roi_seg_neck]
                        self.Roi_sharps = [Roi_sharp]
                        self.Roi_origs = [Roi_orig]
                        if opt.Debug or opt.ReturnImg:
                            self.ims = [im0]
                    else:
                        self.Roi_segs.append(Roi_seg)
                        self.Roi_segs_neck.append(Roi_seg_neck)
                        self.Roi_sharps.append(Roi_sharp)
                        self.Roi_origs.append(Roi_orig)
                        if opt.Debug or opt.ReturnImg:
                            self.ims.append(im0)

                    if img_i == 0:
                        res_FM.append(FM)
                        res_head_coordinates.append(None)
                        res_neck_coordinates.append(None)
                    else:
                        res_FM.append(FM)
                        res_head_coordinates.append(None)
                        res_neck_coordinates.append(None)
                res_sperm_num.append(sperm_num)
                self.img_id += 1

            if cal_detect_time:
                print(f"Detection PostProcess of {opt.batch_size} images:{time.time()-tic_post}")
        return res_head_coordinates, res_neck_coordinates, res_FM, res_sperm_num, file_names

    def segment_sharp(self, res_head_coordinates, res_neck_coordinates, res_FM, res_sperm_num, \
                      file_names, FM_min_idxs):

        FM_min_idxs = FM_min_idxs.tolist()
        res_head_preds = []
        res_vac_preds = []
        res_neck_preds = []
        res_residue_preds = []
        res_area_heads = []
        res_area_vacs = []
        res_focus_pos = []
        Roi_segs = torch.stack([self.Roi_segs[i][0] for i in FM_min_idxs], dim=0).float()
        if TensorRT:
            head_pred, vac_pred = self.unet_model_trt(Roi_segs)
            head_pred = torch.sigmoid(head_pred)
            vac_pred = torch.sigmoid(vac_pred)
        else:
            head_pred, vac_pred = self.unet_HeadVac(Roi_segs)
            head_pred = torch.sigmoid(head_pred)
            vac_pred = torch.sigmoid(vac_pred)
        head_pred[head_pred >= 0.5] = 1
        head_pred[head_pred < 0.5] = 0
        vac_pred[vac_pred >= 0.5] = 1
        vac_pred[vac_pred < 0.5] = 0
        area_head = torch.sum(torch.sum(head_pred, dim=-2), dim=-1)
        area_vac = torch.sum(torch.sum(vac_pred, dim=-2), dim=-1)
        area_head = area_head.squeeze()
        area_vac = area_vac.squeeze()
        # print(area_head, area_vac)

        # neck_residue
        Roi_segs_neck = torch.stack([self.Roi_segs_neck[i][0] for i in FM_min_idxs], dim=0).float()
        if TensorRT:
            neck_pred, residue_pred = self.unet_neck_model_trt(Roi_segs_neck)
            neck_pred = torch.sigmoid(neck_pred)
            residue_pred = torch.sigmoid(residue_pred)
        else:
            neck_pred, residue_pred = self.unet_neck(Roi_segs_neck)
            neck_pred = torch.sigmoid(neck_pred)
            residue_pred = torch.sigmoid(residue_pred)
        neck_pred[neck_pred >= 0.5] = 1
        neck_pred[neck_pred < 0.5] = 0
        residue_pred[residue_pred >= 0.5] = 1
        residue_pred[residue_pred < 0.5] = 0
        neck_pred = neck_pred.squeeze().cpu().numpy().astype(np.uint8)
        residue_pred = residue_pred.squeeze().cpu().numpy().astype(np.uint8)

        b, h, w = neck_pred.shape
        for idx in range(b):
            tmp = get_effective_residue(neck_pred[idx], residue_pred[idx], mini_dist=4)
            if idx == 0:
                residue_pred_ = tmp
            else:
                residue_pred_ = np.concatenate((residue_pred_, tmp), axis=0)

        res_neck_preds.extend(neck_pred)
        res_residue_preds.extend(residue_pred_)

        # area_residue = torch.sum(torch.sum(residue_pred, dim=-2), dim=-1).squeeze()
        # mini_dist = 4
        # neck_new = get_effective_residue(head_pred, vac_pred, mini_dist)

        # Sharp
        Roi_sharps = torch.stack([self.Roi_sharps[i][0] for i in FM_min_idxs], dim=0).float()
        if TensorRT:
            output = self.resnet_model_trt(Roi_sharps)
            output = torch.softmax(output, dim=1)
            cls = torch.argmax(output, dim=1)
            focus_pos = (cls - 20) * 0.5
        else:
            output = self.resnet(Roi_sharps)
            output = torch.softmax(output, dim=1)
            cls = torch.argmax(output, dim=1)
            focus_pos = (cls - 20) * 0.5

        res_head_preds.extend(head_pred)
        res_vac_preds.extend(vac_pred)
        if opt.batch_size == 1:
            res_area_heads.append(area_head)
            res_area_vacs.append(area_vac)
        else:
            res_area_heads.extend(area_head)
            res_area_vacs.extend(area_vac)
        res_focus_pos.extend(focus_pos)
        if opt.Debug or opt.ReturnImg:
            self.res_ims = self.ims

        return res_head_coordinates, res_neck_coordinates, file_names, res_FM, res_focus_pos, \
               res_head_preds, res_vac_preds, res_neck_preds, res_residue_preds, \
               res_area_heads, res_area_vacs, res_sperm_num, FM_min_idxs

    def cal_param(self, res_head_coordinates, res_neck_coordinates, file_names, res_FM, res_focus_pos, \
                  res_head_preds, res_vac_preds, res_neck_preds, res_residue_preds, \
                  res_area_heads, res_area_vacs, res_sperm_num, FM_min_idxs):
        Res_f = pd.DataFrame({
            'Head_coordits': res_head_coordinates,  # 0
            'Neck_coordits': res_neck_coordinates,  # 1
            'idxs': FM_min_idxs,                    # 2
            'FM': res_FM,                           # 3
            'focus_pos': res_focus_pos,             # 4
            'head_preds': res_head_preds,           # 5
            'vac_preds': res_vac_preds,             # 6
            'neck_preds': res_neck_preds,           # 7
            'residue_preds': res_residue_preds,     # 8
            'area_heads': res_area_heads,           # 9
            'area_vacs': res_area_vacs,             # 10
            'sperm_num': res_sperm_num,             # 11
            'imgs_path': [i.split('/')[-1] for i in file_names],                # 12
        })

        if opt.Debug or opt.ReturnImg:
            print("=============")
            Res_f['ims'] = [self.res_ims[idx] for idx in FM_min_idxs]

        Res_f.iloc[:, [2, 3, 4, 9, 10]] = Res_f.iloc[:, [2, 3, 4, 9, 10]].to_numpy().astype(np.float)
        Res_f['focus_pos_abs'] = Res_f['focus_pos'].abs()
        Res_f = Res_f.sort_values(by=['FM', 'focus_pos_abs', 'area_heads'], ascending=[True, True, False])
        Res_f['FM_order'] = range(1, Res_f.shape[0] + 1)
        FM_order_top = Res_f['idxs'][:5]
        """
        Res_f Index(['Head_coordits', 'idxs', 'FM', 'focus_pos', 'head_preds', 'vac_preds',
       'area_heads', 'area_vacs', 'ims', 'focus_pos_abs', 'FM_order'], dtype='object')
        """
        if opt.ShowInfo:
            print(Res_f.columns)
            print(Res_f.iloc[:20, [1, 3, 4, 9, 10]])

        # 后处理，选择清晰图像并统计头部尺寸、空泡信息
        elps = []
        head_size = []
        perimeters = []
        elps_extent = []
        pts_major = []
        pts_minor = []
        pt_major_idxs = []
        vac_params = []
        vac_nums = []
        vac_locs = []
        neck_length = []
        neck_width = []
        residue_area = []
        pts_neck = []
        head_neck_angles = []

        nums = len(res_FM)
        print(f'number of sperm for segmentation:{nums}')
        Res_f.sort_values(by='FM', ascending=True, inplace=True)
        FM_min50 = np.array(Res_f[:][:nums])
        for idx, item in enumerate(FM_min50[:]):
            # head_vac parameters calculation
            head_pred = item[5].float()
            head_pred = head_pred.squeeze().cpu().numpy()
            vac_pred = item[6].float()
            vac_pred = vac_pred.squeeze().cpu().numpy()
            head_coord = item[0]
            neck_coord = item[1]

            if head_coord is not None:
                res = head_MorAnalysis(head_pred)
                ellipse = res[0]
            else:
                ellipse = None

            elps.append(ellipse)
            if ellipse is not None:
                _, (size_b, size_a), _ = ellipse
                size_b = size_b  # * 80 / 1000
                size_a = size_a  # * 80 / 1000
                head_size.append([size_b, size_a])
                pt_major = res[1]  # (pt1, pt2)
                pts_major.append(pt_major)

                pt_minor = res[2]  # (pt3, pt4)
                pts_minor.append(pt_minor)

                perimeter = res[3]
                perimeters.append(perimeter)

                iou = res[4]
                elps_extent.append(iou)

                if neck_coord is not None and head_coord is not None:
                    pt_major_idx, area_d_length_pos, vac_num, vac_loc = vac_MorAnalysis(vac_pred, pt_major, pt_minor, head_coord, neck_coord, ellipse)
                    pt_major_idxs.append([pt_major_idx])
                    vac_params.append(area_d_length_pos)
                    vac_nums.append(vac_num)
                    vac_locs.append(vac_loc)
                else:
                    pt_major_idxs.append([])
                    vac_params.append([])
                    vac_nums.append(0)
                    vac_locs.append(3)
            else:
                head_size.append([0, 0])
                perimeters.append(0)
                elps_extent.append(0)
                pts_major.append((0, 0))
                pts_minor.append((0, 0))
                pt_major_idxs.append([])
                vac_params.append([])
                vac_nums.append(0)
                vac_locs.append(3)

            # neck_residue parameters calculation
            if neck_coord is not None:
                neck_pred = item[7]
                residue_pred = item[8]
                neck_len, neck_wid, neck_points = get_neck_data(neck_pred)
                res_area = np.sum(residue_pred)
                neck_length.append(neck_len)
                neck_width.append(neck_wid)
                residue_area.append(res_area)
                pts_neck.append(neck_points)
            else:
                neck_length.append(0)
                neck_width.append(0)
                residue_area.append(0)
                pts_neck.append((0, 0))

            if np.array(pts_major[-1]).reshape(-1).shape[0] != 2 and np.array(pts_neck[-1]).reshape(-1).shape[0] != 2:
                head_neck_angles.append(cal_angle(pts_major[-1], pts_neck[-1], head_coord, neck_coord, pt_major_idxs[-1][0]))
            else:
                head_neck_angles.append(-1)


        # 精子尺寸统计
        scale_factor = 12.5
        MinAx, MajAx = np.array([i[0] for i in head_size]), np.array([i[1] for i in head_size])
        LenWidRatio = MajAx / (MinAx + 1e-8)
        FM_min50 = Res_f[:][:nums]
        FM_min50['head_size'] = head_size
        FM_min50['MajAx'] = MajAx / scale_factor
        FM_min50['MinAx'] = MinAx / scale_factor
        FM_min50['Perimeter'] = perimeters
        FM_min50['LenWidRatio'] = LenWidRatio
        FM_min50['elps_extent'] = elps_extent
        FM_min50['pts_major'] = pts_major
        FM_min50['pts_minor'] = pts_minor
        FM_min50['vac_params'] = vac_params
        FM_min50['neck_length'] = [i / scale_factor for i in neck_length]
        FM_min50['neck_width'] = [i / scale_factor for i in neck_width]
        FM_min50['residue_area'] = residue_area
        FM_min50['head_neck_angles'] = head_neck_angles
        FM_min50['vac_head_ratio'] = FM_min50['area_vacs'].astype(float) / (FM_min50['area_heads'].astype(float) + 1e-10)
        FM_min50['rsd_head_ratio'] = FM_min50['residue_area'].astype(float) / (FM_min50['area_heads'].astype(float) + 1e-10)
        FM_min50['vac_nums'] = vac_nums
        FM_min50['vac_locs'] = vac_locs


        # 精子大小
        num = 5
        FM_min50.sort_values(by='area_heads', ascending=False, inplace=True)
        FM_min50_tmp = FM_min50[:]
        FM_min50_tmp.sort_values(by='MajAx', ascending=False, inplace=True)
        MajAx_mu = np.mean(np.array(FM_min50_tmp['MajAx'][2:num]))
        MajAx_std = np.std(np.array(FM_min50_tmp['MajAx'][2:num]))

        FM_min50_1 = FM_min50_tmp
        FM_min50_1.sort_values(by='MinAx', ascending=False, inplace=True)
        MinAx_mu = np.mean(np.array(FM_min50_1['MinAx'][2:num]))
        MinAx_std = np.std(np.array(FM_min50_1['MinAx'][2:num]))

        if opt.ShowInfo:
            print('Initial Major Axis:', MajAx_mu, MajAx_std)
            print('Initial Minor Axis:', MinAx_mu, MinAx_std)
            # 显示长轴、短轴
            print('*' * 100)
            print('*' * 100)
            print("Datasets:", img_name)

        # 选出与长、短轴最接近的精子序号
        L1_norm = np.abs(FM_min50_tmp['MinAx'] - MinAx_mu) + np.abs(FM_min50_tmp['MajAx'] - MajAx_mu)
        FM_min50_tmp['L1_norm'] = L1_norm
        FM_min50_tmp.sort_values(by='L1_norm', ascending=True, inplace=True)
        head_area_mu = np.mean(FM_min50_tmp['area_heads'][:5])
        MajAx_mu = np.mean(FM_min50_tmp['MajAx'][2:5])
        MinAx_mu = np.mean(FM_min50_tmp['MinAx'][3:7])
        len_mu = np.mean(FM_min50_tmp['Perimeter'][3:7])
        LenWidRatio_mu = np.mean(FM_min50_tmp['LenWidRatio'][3:7])
        elps_extent_mu = np.mean(FM_min50_tmp['elps_extent'][3:7])

        if opt.ShowInfo:
            print('Sperm Head Informations')
            print('Major Axis:', MajAx_mu, MajAx_std)
            print('Minor Axis:', MinAx_mu, MinAx_std)

        # 显示最接近的的精子序号及面积
        if opt.ShowInfo:
            print('head area:', head_area_mu)
            print('Top5 nearest neighboring sperms for MajAx_mu and MinAx_mu:')
            print(FM_min50_tmp[['idxs', 'head_size', 'area_heads', 'focus_pos']][:5])

        # Sharp_idxs = FM_min50.index[:5]
        Sharp_idxs = FM_order_top

        headsize_idxs = FM_min50_tmp.index[:5]

        seg_name = 'Before2021_WZ20220118'
        if opt.ReturnImg:
            print('Save sharp images')
            for idx in Sharp_idxs:
                idx = int(idx)
                img = self.res_ims[idx]
                idx_ = int(FM_min_idxs.astype(np.int).tolist().index(idx))
                if res_head_coordinates[idx_] is not None:
                    crop_pos = torch.Tensor(res_head_coordinates[idx_])
                    cent_x = torch.clamp((crop_pos[0] + crop_pos[2]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    cent_y = torch.clamp((crop_pos[1] + crop_pos[3]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    head_pred = res_head_preds[idx_].squeeze()
                    vac_pred = res_vac_preds[idx_].squeeze()
                    # print(img.shape, cent_x, cent_y, self.crop_size, head_pred.shape)
                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][
                        head_pred.cpu().numpy() == 1] = (39, 129, 113)
                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][
                        vac_pred.cpu().numpy() == 1] = (0, 0, 255)

                if res_neck_coordinates[idx_] is not None:
                    crop_pos = torch.Tensor(res_neck_coordinates[idx_])
                    cent_x = torch.clamp((crop_pos[0] + crop_pos[2]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    cent_y = torch.clamp((crop_pos[1] + crop_pos[3]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    neck_pred = res_neck_preds[idx_]
                    residue_pred = res_residue_preds[idx_]
                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][
                        neck_pred == 1] = (255, 255, 0)
                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][
                        residue_pred == 1] = (0, 255, 0)

                self.imgs_sharp.append(img)
                idx = '%03d.png' % idx
                f_name = file_names[idx_].split('/')[-1] if '.' in file_names[idx_] else idx
                os.makedirs(f"output/{seg_name}/{img_name}", exist_ok=True)
                cv2.imwrite(f"output/{seg_name}/{img_name}/{f_name}", img)

            FM_min50_tmp.sort_values(by='FM', ascending=True, inplace=True)
            MajAx_mu_ = np.mean(FM_min50_tmp['MajAx'][0:5])
            MinAx_mu_ = np.mean(FM_min50_tmp['MinAx'][0:5])
            idxs = FM_min50_tmp.index[:5]
            tmp_res = [MajAx_mu_, MinAx_mu_]
            tmp_res.extend(idxs)
            np.savetxt(f"output/{seg_name}/{img_name}/res.txt",
                       np.array(tmp_res),
                       fmt='%s',
                       delimiter='\t',
                       newline='\n')

        if opt.ShowInfo:
            print('*' * 50)
            print('Sperm Vacuoles Informations')
        # 选出空泡面积最大的精子
        FM_min50_tmp.sort_values(by='area_vacs', ascending=False, inplace=True)
        vac_area_mu = np.mean(FM_min50_tmp['area_vacs'][:3])
        vac_ratio = vac_area_mu / (head_area_mu + 1E-6)

        if opt.ShowInfo:
            print('vac_area:', vac_area_mu)
            print('vac_ratio:%2.3f' % (vac_ratio * 100.0) + '%')
            print('Top 5 for areas of vacuoles')
            print(FM_min50_tmp[['idxs', 'area_vacs', 'focus_pos']][:5])
            print('*' * 100)
        vac_idxs = FM_min50_tmp.index[:5]

        # 保存全部处理后的图像
        # print(opt.Debug, opt.ReturnImg)
        if opt.Debug and not opt.ReturnImg:
            for idx, img in enumerate(self.res_ims):
                idx_ = int(FM_min_idxs.astype(np.int).tolist().index(idx))
                if res_head_coordinates[idx_] is not None:
                    crop_pos = torch.Tensor(res_head_coordinates[idx_])
                    cent_x = torch.clamp((crop_pos[0] + crop_pos[2]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    cent_y = torch.clamp((crop_pos[1] + crop_pos[3]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    head_pred = res_head_preds[idx_].squeeze()
                    vac_pred = res_vac_preds[idx_].squeeze()
                    # img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    # int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][
                    #     head_pred.cpu().numpy() == 1] = (39, 129, 113)
                    # img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    # int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][
                    #     vac_pred.cpu().numpy() == 1] = (0, 0, 255)
                    # 绘图
                    line_width = 1
                    head_part = img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                                int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :].copy()
                    msk_head = head_pred.cpu().numpy().copy()
                    contours = get_cnt(msk_head)
                    head_part = cv2.drawContours(head_part, contours, -1, (255, 0, 0), line_width)

                    msk_vac = vac_pred.cpu().numpy().copy()
                    contours = get_cnt(msk_vac)
                    head_part = cv2.drawContours(head_part, contours, -1, (0, 0, 255), line_width)

                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :] = head_part

                if res_neck_coordinates[idx_] is not None:
                    crop_pos = torch.Tensor(res_neck_coordinates[idx_])
                    cent_x = torch.clamp((crop_pos[0] + crop_pos[2]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    cent_y = torch.clamp((crop_pos[1] + crop_pos[3]) / 2, self.crop_size // 2,
                                         self.orig_size - self.crop_size // 2).cpu().numpy()
                    neck_pred = res_neck_preds[idx_].squeeze()
                    residue_pred = res_residue_preds[idx_].squeeze()
                    # img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    # int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][neck_pred == 1] = (
                    #     255, 255, 0)
                    # img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    # int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][residue_pred == 1] = (
                    #     0, 255, 0)
                    # 绘图
                    neck_part = img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                                int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :].copy()
                    msk_neck = neck_pred
                    contours = get_cnt(msk_neck)
                    neck_part = cv2.drawContours(neck_part, contours, -1, (255, 255, 0), line_width)

                    msk_residue = residue_pred
                    contours = get_cnt(msk_residue)
                    neck_part = cv2.drawContours(neck_part, contours, -1, (0, 255, 0), line_width)

                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :] = neck_part

                # idx = '%03d.png' % idx
                # f_name = file_names[idx_] if '.' in file_names[idx_] else idx
                # os.makedirs(f"output/{seg_name}/{img_name}", exist_ok=True)
                # cv2.imwrite(f"output/{seg_name}/{img_name}/{f_name}", img)
                filename = file_names[idx_]
                tmp_str = filename.split('/')[-2]
                save_path = filename.replace(tmp_str, 'AI_union_test')
                os.makedirs("/".join(save_path.split('/')[:-1]), exist_ok=True)
                cv2.imwrite(save_path, img)
            self.res_ims = []

        # save to .csv
        # Head_coordits, Neck_coordits, idxs, FM, focus_pos, head_preds, vac_preds, neck_preds, residue_preds, area_heads, area_vacs,
        #       0               1        2     3      4          5            6          7              8           9           10
        # sperm_num, imgs_path, ims, focus_pos_abs, FM_order, head_size, MajAx, MinAx, Perimeter, LenWidRatio, elps_extent, pts_major, pts_minor,
        #      11       12      13        14          15          16       17    18       19           20          21         22         23
        # vac_params, neck_length, neck_width, residue_area, head_neck_angles, vac_head_ratio, rsd_head_ratio, vac_nums, vac_locs, L1_norm
        #     24           25          26          27                28              29               30          31        32        33
        # FM_min50_tmp.sort_values(by='idxs', ascending=True, inplace=True)
        # FM_min50_save = FM_min50_tmp.iloc[:, [12, 2, 11, 0, 17, 18, 9, 19, 20, 21, 3, 4, 15, 22, 23, 29, 1, 10, 24, 31, 32, 25, 26, 27, 28]]
        # print(f"output/{seg_name}/{img_name}")
        # FM_min50_save.to_csv(f"output/{seg_name}/{img_name}/head_vac_res.csv", sep='\t', index=None)
        res_focus_pos = [i.cpu().numpy() for i in res_focus_pos]

        # classification
        FM_min50_tmp.sort_values(by=['FM', 'focus_pos_abs', 'area_heads'], ascending=[True, True, False], inplace=True)
        FM_min5 = FM_min50_tmp.iloc[:5, [12, 17, 18, 20, 29, 31, 32, 26, 28, 30]]
        FM_min5 = FM_min5.round({'MajAx': 2, 'MinAx': 2, 'LenWidRatio':2, 'vac_head_ratio':2, \
                       'vac_nums': 0, 'vac_locs': 0, 'neck_width': 2, 'head_neck_angles': 2, \
                       'rsd_head_ratio': 2})
        cls_res = []
        for i in range(5):
            tmp = FM_min5.iloc[i, 1:].to_numpy().astype(np.float).tolist()
            si = SingleImageConclusion(tmp)
            cls_res.append(si.image_conclusion())
        FM_min5['cls_res'] = cls_res
        # FM_min5.to_csv(f"output/{seg_name}/{img_name}/cls_res.csv", sep='\t', index=None)
        save_path = "/".join(file_names[0].split('/')[:-2])
        FM_min5.to_csv(f"{save_path}/head_vac_res.csv", sep='\t', index=None)

        print('Sharp_idxs:', Sharp_idxs.astype(np.int).tolist())

        if opt.ReturnImg:
            return {'Head_length': float(MajAx_mu),
                    'Head_width': float(MinAx_mu),
                    'Head_area': float(head_area_mu),
                    'Head_arcLen': float(len_mu),
                    'LenWidRatio': float(LenWidRatio_mu),
                    'elps_extent': float(elps_extent_mu),
                    'Vac_area': float(vac_area_mu),
                    'Sharp_idxs': Sharp_idxs.tolist(),
                    'Sharp_info': [float(np.mean(res_focus_pos)), float(np.std(res_focus_pos)),
                                   float(np.max(res_focus_pos)), float(np.min(res_focus_pos))],
                    'sperm_num': [float(i) for i in res_sperm_num.tolist()],
                    'Sharp_imgs': self.imgs_sharp}
        else:
            return {'Head_length': float(MajAx_mu),
                    'Head_width': float(MinAx_mu),
                    'Head_area': float(head_area_mu),
                    'Head_arcLen': float(len_mu),
                    'LenWidRatio': float(LenWidRatio_mu),
                    'elps_extent': float(elps_extent_mu),
                    'Vac_area': float(vac_area_mu),
                    'Sharp_idxs': Sharp_idxs.tolist(),
                    'Sharp_info': [np.mean(res_focus_pos), np.std(res_focus_pos), np.max(res_focus_pos),
                                   np.min(res_focus_pos)]}

        # Write_Text('temp.txt', f"{MajAx_mu}\n{MinAx_mu}\n{head_area_mu}\n{vac_area_mu}\n{headsize_idxs.tolist()}\n{vac_idxs.tolist()}\n{[np.mean(res_focus_pos), np.std(res_focus_pos), np.max(res_focus_pos), np.min(res_focus_pos)]}\n{focus_abs}]")

        if opt.Debug:
            return results, headsize_idxs
        else:
            return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/custom/s12-1/*.*", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny_headneckImpurity_anchorUpdate_v1.cfg",
                        help="path to model definition file")
    parser.add_argument("--class_path", type=str, default="weights/YOLOv3_S11_0330_0418/classes.names",
                        help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--FM_min_num", type=int, default=30, help="size of selected images for segment")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--yolo_img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--unet_img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--resnet_img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--TensorRT", action='store_true', default=False, help="whether use TensorRT")
    parser.add_argument("--Half", action='store_true', default=False, help="whether use Half")
    parser.add_argument("--Speed_Test", action='store_true', default=False, help="whether use Half")
    parser.add_argument("--BiLinear", action='store_true', default=True, help="whether use Half")
    parser.add_argument("--Debug", action='store_true', default=False, help="Debug mode")
    parser.add_argument("--ShowInfo", action='store_true', default=False, help="Show mode")
    parser.add_argument("--ReturnImg", action='store_true', default=False, help="Show mode")
    parser.add_argument("--yolo_weights_path", type=str, default="weights/YOLOv3_S11_0330_0418/yolov3_ckpt_f1_best.pth",
                        help="path to weights file")
    parser.add_argument("--unet_weights_path", type=str,
                        default="weights/Before2021_WZ20220118_20220309vac_20220427vac_Unet_BestHead.pth",
                        help="path to weights file")
    parser.add_argument("--unet_neck_weights_path", type=str, default="weights/Neck_0415_0518_UNet_Pytorch140.pth",
                        help="path to weights file")
    parser.add_argument("--resnet_weights_path", type=str,
                        default="weights/resnet34_sharp_join_Plateau_lr1E-3_bs16_WeiClip/model_acc2.pth",
                        help="path to weights file")
    parser.add_argument('--arch', default='resnet34',
                        help='model architecture')
    opt = parser.parse_args()
    print(opt)
    img_name = opt.image_folder.split('/')[-2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)
    cal_time = True
    cal_detect_time = False

    # model selection
    # TensorRT: YOLO and FPN can be accerlated, YOLO head not
    Speed_Test = opt.Speed_Test
    TensorRT = opt.TensorRT
    Half = opt.Half  # half precision
    test_flag = False

    model = Detect()
    with torch.no_grad():
        img_dirs = glob.glob(opt.image_folder)
        img_dirs = sorted(img_dirs)
        opt.FM_min_num = min(opt.FM_min_num, len(img_dirs))
        print(f"Number of images:{len(img_dirs)}")
        # print(img_dirs)
        return_num = 9
        for i in range(1):
            if test_flag:
                detect_pred = []
            else:
                detect_pred = [[] for i in range(5)]
            bs = opt.batch_size

            tic_detect = time.time()
            for idx, img_dir in enumerate(img_dirs):
                img = cv2.imread(img_dir, -1)
                if idx % bs == 0:
                    imgs = [img]
                else:
                    imgs.append(img)

                if idx % bs == bs - 1 or idx == len(img_dirs) - 1:
                    imgs = np.stack(imgs, axis=0)
                    input_imgs, orig_imgs = model.pre_process(imgs)
                    if test_flag:
                        temp = model.predict_FM(input_imgs, orig_imgs)
                        detect_pred.extend(temp)
                    else:
                        temp = model.predict(input_imgs, orig_imgs)
                        for i in range(5):
                            detect_pred[i].extend(temp[i])

            detect_pred = np.array(detect_pred, dtype=object)
            print(detect_pred.shape, len(img_dirs))
            detect_pred[4, :] = [i for i in img_dirs]    # filenames
            if opt.Debug:
                opt.FM_min_num = detect_pred.shape[1]
            FM_min_idxs = np.argsort(detect_pred[2, :])[:opt.FM_min_num]
            detect_pred = detect_pred[:, FM_min_idxs]
            detect_pred = np.concatenate((detect_pred, FM_min_idxs.reshape(1, -1)), axis=0)
            if cal_time:
                print(f'Detection used time:{time.time()-tic_detect}')
                tic_seg = time.time()

            # 计算分割结果
            res_pred = [[] for i in range(13)]
            for start_i in range(0, opt.FM_min_num, bs):
                end_i = min(opt.FM_min_num, start_i + bs)
                bs_detect = np.array(detect_pred[:, start_i:end_i])
                temp = model.segment_sharp(
                    bs_detect[0, :],
                    bs_detect[1, :],
                    bs_detect[2, :],
                    bs_detect[3, :],
                    bs_detect[4, :],
                    bs_detect[5, :],
                )
                for i in range(13):
                    res_pred[i].extend(temp[i])
            # res_pred = np.array(res_pred)   #, dtype=object
            if cal_time:
                print(f'Segmentation used time:{time.time() - tic_seg}')
                tic_cal = time.time()

            if test_flag:
                out = [i.cpu().numpy() for i in res_pred]
            else:
                out = model.cal_param(res_pred[0],
                                      res_pred[1],
                                      res_pred[2],
                                      res_pred[3],
                                      [i.cpu() for i in res_pred[4]],
                                      [i.cpu() for i in res_pred[5]],
                                      [i.cpu() for i in res_pred[6]],
                                      res_pred[7],
                                      res_pred[8],
                                      [i.cpu() for i in res_pred[9]],
                                      [i.cpu() for i in res_pred[10]],
                                      res_pred[11],
                                      res_pred[12],
                                      )
                # out = model.cal_param(res_pred[0, :],
                #                       res_pred[1, :],
                #                       res_pred[2, :],
                #                       res_pred[3, :],
                #                       res_pred[4, :],
                #                       res_pred[5, :],
                #                       res_pred[6, :],
                #                       res_pred[7, :],
                #                       res_pred[8, :],
                #                       res_pred[9, :],
                #                       res_pred[10, :],
                #                       res_pred[11, :],
                #                       res_pred[12, :],
                #                       )
            # print(out)
            if cal_time:
                print(f'Calculation parameters used time: {time.time() - tic_cal}')
                print(f'Total used time: {time.time() - tic_detect}')
            model.reset()
