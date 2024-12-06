from __future__ import division

from models import *
from Yolov3_utils.utils import *
from Yolov3_utils.datasets import *

import os
import glob
import sys
import time
import argparse

import torch.nn.functional as F
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from torch2trt import torch2trt, TRTModule
from Unet_unet import UNet
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
from utils import head_MorAnalysis, vac_MorAnalysis, bbox_iou, \
    get_effective_residue, SingleImageConclusion, get_neck_data, cal_angle

# STMClient
from stm_worker2 import STMWorker, BufferWorker, BatchWorker
import torch.multiprocessing as mp
import json
from tqdm import tqdm
from datetime import datetime
import h5py
import redis
import traceback

STATUS_CLOSED      = 'closed'
STATUS_INIT        = 'initializing'
STATUS_CONNECTED   = 'connected'
STATUS_NETERROR    = 'net_error'
STATUS_ERROR       = 'error'


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
            self.yolo_head = YOLOHead(config_path=opt.model_def)

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

    def first_run(self):
        # 第一次运行速度慢，先运行10次
        batch_size = opt.batch_size
        for i in range(1):
            if TensorRT:
                # yolo
                if Half:
                    test_data = torch.rand(size=(batch_size, 1, opt.yolo_img_size, opt.yolo_img_size)).cuda().half()
                else:
                    test_data = torch.rand(size=(batch_size, 1, opt.yolo_img_size, opt.yolo_img_size)).cuda()
                detections = self.yolo_head(self.yolo_model_trt(test_data))
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
            img, _ = pad_to_square(img, 0)
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
        res_head_preds = []
        res_vac_preds = []
        res_FM = []
        res_area_heads = []
        res_area_vacs = []
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
            if TensorRT:
                detections = self.yolo_head(self.yolo_model_trt(input_imgs), Half=opt.Half)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)
            else:
                detections = self.yolo(input_imgs)
                detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, method=1)

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
                    if opt.Debug or opt.ReturnImg:
                        if head_coordinate is not None:
                            label = '%s %.2f' % (self.classes[int(0)], conf)
                            plot_one_box(head_coordinate, im0, label=label, color=colors[int(0)])

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

                    if opt.Debug or opt.ReturnImg:
                        if neck_coordinate is not None:
                            label = '%s %.2f' % (self.classes[int(1)], conf)
                            plot_one_box(neck_coordinate, im0, label=label, color=colors[int(1)])

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
        # print(f'head seg shape:{Roi_segs.shape}')
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
        # print(f'neck seg shape:{Roi_segs_neck.shape}')
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
        # print(f'head sharp shape:{Roi_sharps.shape}')
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

        res_head_preds.extend(head_pred.cpu())
        res_vac_preds.extend(vac_pred.cpu())
        if opt.batch_size == 1:
            res_area_heads.append(area_head.cpu())
            res_area_vacs.append(area_vac.cpu())
        else:
            res_area_heads.extend(area_head.cpu())
            res_area_vacs.extend(area_vac.cpu())
        res_focus_pos.extend(focus_pos.cpu())
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
            'imgs_path': file_names,                # 12
        })

        if opt.Debug or opt.ReturnImg:
            print("=============")
            # Res_f['ims'] = [self.res_ims[idx] for idx in FM_min_idxs]
            Res_f['ims'] = [None for idx in FM_min_idxs]

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

        Sharp_idxs = FM_order_top

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
                idx = '%03d' % idx
                print(f"output/{seg_name}/{img_name}")
                os.makedirs(f"output/{seg_name}/{img_name}", exist_ok=True)
                cv2.imwrite(f"output/{seg_name}/{img_name}/{idx}.png", img)

        # 保存全部处理后的图像
        # print(opt.Debug, opt.ReturnImg)
        if opt.Debug and not opt.ReturnImg:
            final_result = []
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
                    neck_pred = res_neck_preds[idx_].squeeze()
                    residue_pred = res_residue_preds[idx_].squeeze()
                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][neck_pred == 1] = (
                        255, 255, 0)
                    img[int(cent_y - self.crop_size / 2): int(cent_y + self.crop_size / 2),
                    int(cent_x - self.crop_size / 2): int(cent_x + self.crop_size / 2), :][residue_pred == 1] = (
                        0, 255, 0)
                final_result.append(img)

            final_result = np.array(final_result)
            now = datetime.now()
            dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
            h5file = h5py.File(f"output/{dt_string}.h5", "w")
            img_data = h5file.create_dataset("images", np.shape(final_result), h5py.h5t.STD_U8BE, data=final_result)
            h5file.close()

        # save to .csv
        # Head_coordits, Neck_coordits, idxs, FM, focus_pos, head_preds, vac_preds, neck_preds, residue_preds, area_heads, area_vacs,
        #       0               1        2     3      4          5            6          7              8           9           10
        # sperm_num, imgs_path, ims, focus_pos_abs, FM_order, head_size, MajAx, MinAx, Perimeter, LenWidRatio, elps_extent, pts_major, pts_minor,
        #      11       12      13        14          15          16       17    18       19           20          21         22         23
        # vac_params, neck_length, neck_width, residue_area, head_neck_angles, vac_head_ratio, rsd_head_ratio, vac_nums, vac_locs, L1_norm
        #     24           25          26          27                28              29               30          31        32        33
        FM_min50.sort_values(by=['FM', 'focus_pos_abs', 'area_heads'], ascending=[True, True, False], inplace=True)
        FM_min5 = FM_min50.iloc[:5, [12, 17, 18, 20, 29, 31, 32, 26, 28, 30]]
        FM_min5 = FM_min5.round({'MajAx': 2, 'MinAx': 2, 'LenWidRatio': 2, 'vac_head_ratio': 2, \
                                 'vac_nums': 0, 'vac_locs': 0, 'neck_width': 2, 'head_neck_angles': 2, \
                                 'rsd_head_ratio': 2})
        cls_res = []
        for i in range(5):
            tmp = FM_min5.iloc[i, 1:].to_numpy().astype(np.float).tolist()
            si = SingleImageConclusion(tmp)
            cls_res.append(''.join(si.image_conclusion()))
        FM_min5['cls_res'] = cls_res

        # 根据5张精子图像判断整体状态
        def determine_cls(single_reses):
            #     print(single_reses)
            if len(single_reses) < 5 or single_reses.count('u') >= 3:
                return 'u'
            if single_reses.count('a') >= 3:
                return 'a'
            elif single_reses.count('p') >= 3:
                return 'p'
            elif single_reses.count('n') >= 3:
                return 'n'
            else:
                return 'u'

        img_res = [i[-1] for i in cls_res]
        img5_res = determine_cls(img_res)
        # 正常：s1, 异常：s6，偏正常：s7，不确定：s8
        res2flag = {
            "n": 's1',
            'a': 's6',
            'p': 's7',
            'u': 's8'
        }
        if img5_res in res2flag:
            tmp_name = res2flag[img5_res]
        else:
            tmp_name = 's8'

        if opt.ReturnImg:
            return {
                f'{tmp_name}': [str(i) for i in FM_min5.iloc[0, :].to_numpy().tolist()],
                f's2': [str(i) for i in FM_min5.iloc[1, :].to_numpy().tolist()],
                f's3': [str(i) for i in FM_min5.iloc[2, :].to_numpy().tolist()],
                f's4': [str(i) for i in FM_min5.iloc[3, :].to_numpy().tolist()],
                f's5': [str(i) for i in FM_min5.iloc[4, :].to_numpy().tolist()],
                'Sharp_imgs': self.imgs_sharp,
            }
        else:
            return {
                f's1': [str(i) for i in FM_min5.iloc[0, :].to_numpy().tolist()],
                f's2': [str(i) for i in FM_min5.iloc[1, :].to_numpy().tolist()],
                f's3': [str(i) for i in FM_min5.iloc[2, :].to_numpy().tolist()],
                f's4': [str(i) for i in FM_min5.iloc[3, :].to_numpy().tolist()],
                f's5': [str(i) for i in FM_min5.iloc[4, :].to_numpy().tolist()],
            }

    def reset(self):
        if opt.Debug or opt.ReturnImg:
            self.res_ims = []
            self.imgs_sharp = []
        self.img_id = 0

class PreWorker(mp.Process):
    def __init__(self, batch_queue, tensor_queue, stop_queue, img_size, clear_flag, pre_processing):
        super(PreWorker, self).__init__()
        self.batch_queue = batch_queue
        self.tensor_queue = tensor_queue
        self.stop_queue = stop_queue
        self.img_size = img_size
        self.clear_flag = clear_flag
        self.pre_processing = pre_processing

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
            img, _ = pad_to_square(img, 0)
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
        return (input_imgs, orig_imgs)

    def run(self):
        while not self.stop_queue.empty():
            if self.clear_flag.value > 0:
                while not self.tensor_queue.empty():
                    try:
                        self.tensor_queue.get_nowait()
                    except:
                        pass
                while not self.batch_queue.empty():
                    try:
                        self.batch_queue.get_nowait()
                    except:
                        pass
                continue
            try:
                imgs = self.batch_queue.get_nowait()
                if imgs is not None:
                    # with self.pre_processing.get_lock():
                        # self.pre_processing.value = 1
                    result = self.pre_process(imgs)

                    if self.clear_flag.value == 0:
                        self.tensor_queue.put(result)
                    else:
                        while not self.tensor_queue.empty():
                            try:
                                self.tensor_queue.get_nowait()
                            except:
                                pass

                    with self.pre_processing.get_lock():
                        self.pre_processing.value = 0
            except:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/custom/s12-1/*.*", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny_headneckImpurity_anchorUpdate_v1.cfg",
                        help="path to model definition file")
    parser.add_argument("--class_path", type=str, default="weights/YOLOv3_S11_0330_0418/classes.names",
                        help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=25, help="size of the batches")
    parser.add_argument("--FM_min_num", type=int, default=25, help="size of selected images for segment")
    parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--yolo_img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--unet_img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--resnet_img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--TensorRT", action='store_true', default=True, help="whether use TensorRT")
    parser.add_argument("--Half", action='store_true', default=True, help="whether use Half")
    parser.add_argument("--Speed_Test", action='store_true', default=False, help="whether use Half")
    parser.add_argument("--BiLinear", action='store_true', default=True, help="whether use Half")
    parser.add_argument("--Debug", action='store_true', default=False, help="Debug mode")
    parser.add_argument("--ReturnImg", action='store_true', default=True, help="Show mode")
    parser.add_argument("--ShowInfo", action='store_true', default=False, help="Show mode")
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

    parser.add_argument("--host", type=str, default="192.168.31.47", help="host ip")
    parser.add_argument("--port", type=int, default=6666, help="host port")
    parser.add_argument("--patch_len", type=int, default=400, help="frame_len")

    opt = parser.parse_args()
    # print(opt)
    img_name = opt.image_folder.split('/')[-2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # model selection
    # TensorRT: YOLO and FPN can be accerlated, YOLO head not
    Speed_Test = opt.Speed_Test
    TensorRT = opt.TensorRT
    Half = opt.Half    # half precision

    host = opt.host
    port = opt.port

    model = Detect()

    redis_pool = redis.ConnectionPool(host='172.17.0.1', port=6379, decode_responses=True)
    app_redis = redis.Redis(connection_pool=redis_pool)

    with torch.no_grad():
        mp.set_start_method('spawn')

        buffer_queue = mp.Queue()
        msg_queue = mp.Queue()
        img_queue = mp.Queue()
        out_queue = mp.Queue()
        batch_queue = mp.Queue()
        tensor_queue = mp.Queue()
        image_out_queue = mp.Queue()
        # queue is process safe
        stop_queue = mp.Queue()
        stop_queue.put(0)

        # clearing flag
        clear_flag = mp.Value('i', 0)
        pre_processing = mp.Value('i', 0)

        waiting_array = []
        waiting_result = []

        print(host, port)

        batch_worker = BatchWorker(img_queue, batch_queue, opt.batch_size, stop_queue, clear_flag)
        batch_worker.start()

        buffer_worker = BufferWorker(buffer_queue, img_queue, out_queue, stop_queue, clear_flag)
        buffer_worker.start()

        pre_worker = PreWorker(batch_queue, tensor_queue, stop_queue, opt.yolo_img_size, clear_flag, pre_processing)
        pre_worker.start()

        worker = STMWorker(host, port, buffer_queue, out_queue, stop_queue)
        worker.start()
        app_redis.set('status', STATUS_CONNECTED)

        pbar = None
        tmp = 0
        while not stop_queue.empty():
            if clear_flag.value > 0:
                waiting_result = []
                model.reset()
                if pbar:
                    pbar.close()
                pbar = None
                if img_queue.empty() \
                        and batch_queue.empty() \
                        and tensor_queue.empty() \
                        and pre_processing.value == 0 \
                        and img_queue.qsize() == 0 \
                        and batch_queue.qsize() == 0 \
                        and tensor_queue.qsize() == 0:
                    with clear_flag.get_lock():
                        clear_flag.value = 0
                    print("<<<<<<<<<<<<<<<<clear flag finished")
                else:
                    continue
            try:
                result = tensor_queue.get_nowait()
                if result is not None:
                    if not pbar:
                        pbar = tqdm(total=opt.patch_len)
                        s = time.time()
                    # ss = time.time()
                    # input_imgs, orig_imgs = model.pre_process(imgs)
                    # print(f"preprocessing time: {(time.time() -ss) * 1000}ms")

                    input_imgs = result[0]
                    orig_imgs = result[1]
                    # print(input_imgs.shape)
                    temp = model.predict(input_imgs, orig_imgs)

                    waiting_result.append(np.array(temp).transpose())
                    # print(f"processed: {int(len(waiting_result) / (opt.patch_len / opt.batch_size) * 100)}")
                    pbar.update(opt.batch_size)

                    if len(waiting_result) >= int(opt.patch_len / opt.batch_size):

                        tmp = 1
                        detect_pred = np.vstack(waiting_result)
                        waiting_result = []

                        detect_pred = detect_pred.transpose()
                        detect_pred[4, :] = np.arange(opt.patch_len)  # filenames
                        FM_min_idxs = np.argsort(detect_pred[2, :])[:opt.FM_min_num]
                        detect_pred = detect_pred[:, FM_min_idxs]
                        detect_pred = np.concatenate((detect_pred, FM_min_idxs.reshape(1, -1)), axis=0)

                        print('Detect Finished!')
                        # 计算分割结果
                        res_pred = [[] for i in range(13)]
                        for start_i in range(0, opt.FM_min_num, opt.batch_size):
                            end_i = min(opt.FM_min_num, start_i + opt.batch_size)
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
                        res_pred = np.array(res_pred)

                        print('Segmentation Finished!')
                        # try:
                        out = model.cal_param(
                            res_pred[0, :],
                            res_pred[1, :],
                            res_pred[2, :],
                            res_pred[3, :],
                            res_pred[4, :],
                            res_pred[5, :],
                            res_pred[6, :],
                            res_pred[7, :],
                            res_pred[8, :],
                            res_pred[9, :],
                            res_pred[10, :],
                            res_pred[11, :],
                            res_pred[12, :],
                        )

                        print('Calculate parameters Finished!')
                        if pbar:
                            pbar.close()
                        pbar = None

                        e = time.time()
                        print(f"400 images time:{(e-s) * 1000}ms")
                        if opt.ReturnImg:
                            return_imgs = out.pop('Sharp_imgs')
                            print(f"----->return {len(return_imgs)} imgs")

                        print(out)

                        out_string = json.dumps(out)
                        string_msg = out_string.encode()
                        meta_id = 4
                        meta_id_bytes = meta_id.to_bytes(2, 'big')
                        data_len = len(string_msg) + 2
                        data_len_bytes = data_len.to_bytes(4, 'big')
                        out_buffer = data_len_bytes + meta_id_bytes + string_msg
                        out_queue.put(out_buffer)
                        if opt.ReturnImg:
                            ss = time.time()
                            for a in return_imgs:
                                img_encode = cv2.imencode('.jpg', np.array(a).astype(np.uint8))[1]
                                img_bytes = np.array(img_encode).tobytes()
                                meta_id = 5
                                meta_id_bytes = meta_id.to_bytes(2, 'big')
                                data_len = len(img_bytes) + 2
                                data_len_bytes = data_len.to_bytes(4, 'big')
                                out_buffer = data_len_bytes + meta_id_bytes + img_bytes
                                out_queue.put(out_buffer)
                            ee = time.time()
                            print(f"send image time:{(ee-ss) * 1000}ms")
                        # except:
                        #     print("error cal param")
                        print(f'Total time used: {time.time()-s}')
                else:
                    print("cannot handle a batch")
            except:
                if tmp == 1:
                    traceback.print_exc()
                    tmp = 0


        if pbar:
            pbar.close()

        img_queue.cancel_join_thread()
        batch_queue.cancel_join_thread()
        buffer_queue.cancel_join_thread()
        out_queue.cancel_join_thread()
        stop_queue.cancel_join_thread()

        status = app_redis.get('status')
        if status != STATUS_NETERROR:
            app_redis.set('status', STATUS_CLOSED)

        print('-----work join---')
        worker.join()
        print('-----batch_worker join---')
        batch_worker.join()
        print('-----buffer_worker join---')
        buffer_worker.join()
        print('-----pre_worker join---')
        pre_worker.join()


