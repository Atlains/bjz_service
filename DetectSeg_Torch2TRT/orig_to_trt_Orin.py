from __future__ import division

from models import *
import argparse

import torch
from torch2trt import torch2trt
from Unet_unet import UNet
import torchvision.models as models
import torch.nn as nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/custom/s12-1/", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
    parser.add_argument("--class_path", type=str, default="data/custom/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--yolo_img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--unet_img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--resnet_img_size", type=int, default=128, help="size of each image dimension")
    parser.add_argument("--max_batch_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--TensorRT", action='store_true', default=False, help="whether use TensorRT")
    parser.add_argument("--Half", action='store_true', default=False, help="whether use Half")
    parser.add_argument("--Speed_Test", action='store_true', default=False, help="whether use Half")
    parser.add_argument("--BiLinear", action='store_true', default=False, help="whether use Half")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    parser.add_argument("--yolo_weights_path", type=str, default="weights/YOLOv3_S11_416_new/yolov3_ckpt_380.pth", help="path to weights file")
    parser.add_argument("--unet_weights_path", type=str, default="weights/Before2021_WZ20220118_20220309vac_20220427vac_Unet_BestHead.pth", help="path to weights file")
    parser.add_argument("--unet_neck_weights_path", type=str, default="weights/Neck_0415_0518_UNet_Pytorch140.pth", help="path to weights file")
    # parser.add_argument("--unet_weights_path", type=str, default="weights/WZ20211125_UNet_BestHead.pth", help="path to weights file")
    # parser.add_argument("--unet_weights_path", type=str, default="weights/Unet_epoch281.pth", help="path to weights file")
    parser.add_argument("--resnet_weights_path", type=str, default="weights/resnet34_sharp_join_Plateau_lr1E-3_bs16_WeiClip/model_acc2.pth", help="path to weights file")
    parser.add_argument('--arch', default='resnet34',
                        # choices=res2net.__all__,
                        help='model architecture')
    opt = parser.parse_args()
    # print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model selection
    # TensorRT: YOLO and FPN can be accerlated, YOLO head not
    TensorRT = opt.TensorRT
    Half = opt.Half    # half precision

    # create model
    num_classes = 41
    resnet = models.__dict__[opt.arch]()
    resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = resnet.fc.in_features
    resnet.fc = nn.Linear(fc_features, num_classes)

    if TensorRT is True:
        if Half is True:
            yolo_model_backbone = Darknet_Backbone(opt.model_def, img_size=opt.yolo_img_size).to(device).half()
            unet_HeadVac = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device).half()
            unet_neck = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device).half()
            resnet = resnet.to(device).half()
        else:
            yolo_model_backbone = Darknet_Backbone(opt.model_def, img_size=opt.yolo_img_size).to(device)
            unet_HeadVac = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device)
            unet_neck = UNet(n_channels=1, head_classes=1, vac_classes=1, bilinear=opt.BiLinear).to(device)
            resnet = resnet.to(device)

        # Load checkpoint weights
        yolo_model_backbone.load_state_dict(torch.load(opt.yolo_weights_path, map_location=device))
        unet_HeadVac.load_state_dict(torch.load(opt.unet_weights_path, map_location=device))
        unet_neck.load_state_dict(torch.load(opt.unet_neck_weights_path, map_location=device))
        resnet.load_state_dict(torch.load(opt.resnet_weights_path, map_location=device))

        # Set in evaluation mode: when forward pass, BatchNormalization and Dropout will be ignored
        yolo_model_backbone.eval()
        unet_HeadVac.eval()
        unet_neck.eval()
        resnet.eval()

        # Add Detection Head
        yolo_head = YOLOHead(config_path=opt.model_def)

        # DarknetBackbone convert to TensorRT
        if Half is True:
            # yolo
            x = torch.rand(size=(opt.batch_size, 1,  opt.yolo_img_size, opt.yolo_img_size)).cuda().half()
            yolo_model_trt = torch2trt(yolo_model_backbone, [x], fp16_mode=True, max_batch_size=opt.max_batch_size)
            torch.save(yolo_model_trt.state_dict(), opt.yolo_weights_path.split('.')[0] + f'_Half_trt_bs{opt.batch_size}.pth')

            # unet_headvac
            x = torch.rand(size=(opt.batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda().half()
            unet_model_trt = torch2trt(unet_HeadVac, [x], fp16_mode=True, max_batch_size=opt.max_batch_size)
            torch.save(unet_model_trt.state_dict(), opt.unet_weights_path.split('.')[0] + f'_Half_trt_bs{opt.batch_size}.pth')

            # unet_neck
            x = torch.rand(size=(opt.batch_size, 1, opt.unet_img_size, opt.unet_img_size)).cuda().half()
            unet_neck_model_trt = torch2trt(unet_neck, [x], fp16_mode=True, max_batch_size=opt.max_batch_size)
            torch.save(unet_neck_model_trt.state_dict(), opt.unet_neck_weights_path.split('.')[0] + f'_Half_trt_bs{opt.batch_size}.pth')

            # resnet
            x = torch.rand(size=(opt.batch_size, 1, opt.resnet_img_size, opt.resnet_img_size)).cuda().half()
            resnet_model_trt = torch2trt(resnet, [x], fp16_mode=True, max_batch_size=opt.max_batch_size)
            torch.save(resnet_model_trt.state_dict(), opt.resnet_weights_path.split('.')[0] + f'_Half_trt_bs{opt.batch_size}.pth')
        else:
            # yolo
            x = torch.rand(size=(opt.batch_size, 1,  opt.yolo_img_size, opt.yolo_img_size)).cuda()
            yolo_model_trt = torch2trt(yolo_model_backbone, [x], max_batch_size=opt.max_batch_size)
            torch.save(yolo_model_trt.state_dict(), opt.yolo_weights_path.split('.')[0] + f'_trt_bs{opt.batch_size}.pth')

            # uent
            x = torch.rand(size=(opt.batch_size, 1,  opt.unet_img_size, opt.unet_img_size)).cuda()
            unet_model_trt = torch2trt(unet_HeadVac, [x], max_batch_size=opt.max_batch_size)
            torch.save(unet_model_trt.state_dict(), opt.unet_weights_path.split('.')[0] + f'_trt_bs{opt.batch_size}.pth')

            # uent_neck
            x = torch.rand(size=(opt.batch_size, 1,  opt.unet_img_size, opt.unet_img_size)).cuda()
            unet_neck_model_trt = torch2trt(unet_neck, [x], max_batch_size=opt.max_batch_size)
            torch.save(unet_neck_model_trt.state_dict(), opt.unet_neck_weights_path.split('.')[0] + f'_trt_bs{opt.batch_size}.pth')

            # resnet
            x = torch.rand(size=(opt.batch_size, 1, opt.resnet_img_size, opt.resnet_img_size)).cuda()
            resnet_model_trt = torch2trt(resnet, [x], max_batch_size=opt.max_batch_size)
            torch.save(resnet_model_trt.state_dict(), opt.resnet_weights_path.split('.')[0] + f'_trt_bs{opt.batch_size}.pth')
