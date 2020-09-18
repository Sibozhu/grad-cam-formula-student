#!/usr/bin/env python
from __future__ import print_function

import copy

import click
import cv2
import numpy as np
import torch
import os, os.path
import datetime
from torch.autograd import Variable
from torchvision import models, transforms
from PIL import Image, ImageDraw

from src.utils.models import Darknet
from src.utils.grad_cam import (BackPropagation, Deconvolution, GradCAM, GuidedBackPropagation)

def grad_save(filename, flag_data):
    flag_data -= flag_data.min()
    flag_data /= flag_data.max()
    flag_data *= 255.0

    cv2.imwrite(filename, np.uint8(flag_data))


def grad_cam_save(filename, flag_data, raw_img):
    height, width, _ = raw_img.shape
    flag_data = cv2.resize(flag_data, (width, height))
    flag_data = cv2.applyColorMap(np.uint8(flag_data * 255.0), cv2.COLORMAP_JET)
    # flag_data = flag_data.astype(np.float) + raw_img.astype(np.float)
    flag_data = flag_data / flag_data.max() * 255.0
    cv2.imwrite(filename, np.uint8(flag_data))

with open('./src/config/intersting_layers.txt', 'r') as f:
    intersting_layer = f.read().split('\n')
print('GRAD-CAM Visualization on: '+str(intersting_layer))

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.append('yolov3')

@click.command()
@click.option('-i', '--image-directory-path', help="Path to image directory", type=str, required=True)
@click.option('-a', '--arch', help="a model name from ```torchvision.models```, e.g., 'resnet152' (default: yolov3)", type=click.Choice(model_names), default='yolov3')
@click.option('-n', '--num_class', help="number of classes to generate (default: 1)", type=int, default=1)
@click.option('--cuda/--no-cuda', help="GPU mode or CPU mode (default: cuda mode)", default=True)
def main(image_directory_path, arch, num_class, cuda):

    startTime = datetime.datetime.now()

    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.36',
            'input_size': 224
        },
        'vgg19_bn': {
            'target_layer': 'features.52',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
        'densenet201': {
            'target_layer': 'features.denseblock4',
            'input_size': 224
        },
        'yolov3': {
            'target_layer': 'module_list.5.conv_5',
            'input_size': 544
            # 'input_size': 700
            # 'input_size': 416
        },
        # Adding model
    }.get(arch)

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    # dictionary = list()
    # with open('src/utils/synset_words.txt') as lines:
    #     for line in lines:
    #         line = line.strip().split(' ', 1)[1]
    #         line = line.split(', ', 1)[0].replace(' ', '_')
    #         dictionary.append(line)

    model_cfg = "./src/config/yolov3.cfg"
    weights_path = "./src/config/yolov3.weights"
    model = Darknet(model_cfg, CONFIG['input_size'])
    model.load_weights(weights_path)

    # model = models.__dict__[arch](pretrained=True)
    # print(*list(model.named_modules()), sep='\n')

    model.to(device)
    model.eval()

    img_path = image_directory_path
    ttl_img = []
    valid_images = [".jpg"]
    for f in os.listdir(img_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        ttl_img.append((os.path.join(img_path, f)))
    # print(ttl_img)
    print(str(len(ttl_img)) + " images in total") 

    for k in ttl_img:    

        flag_img_name= (k.split("/"))
        img_name = flag_img_name[-1]
        del flag_img_name
        print('loading '+img_name+' ......')

        raw_img = Image.open(k).convert('RGB')
        # w, h = CONFIG['input_size'],CONFIG['input_size']
        w,h = raw_img.width,raw_img.height
        # print(raw_img.size)

        max_dimension = max(h, w)
        pad_w = int((max_dimension - w) / 2)
        pad_h = int((max_dimension - h) / 2)
        # ratio = float(CONFIG['input_size']) / float(max_dimension)
        del max_dimension

        raw_img = transforms.functional.pad(raw_img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        # raw_img = transforms.functional.resize(raw_img, (CONFIG['input_size'], CONFIG['input_size']))
        # print(raw_img.size)
        raw_img = transforms.functional.resize(raw_img, (w, w))   
        raw_img = transforms.functional.to_tensor(raw_img)


        image = raw_img.unsqueeze(0)
        raw_img = raw_img.permute(2, 1, 0)
        print('loading complete!')
        # =========================================================================

        print('Grad-CAM Visualization on '+img_name)

        grad_cam = GradCAM(model=model)
        probs, idx = grad_cam.forward(image.to(device))
        del image

        for i in range(0,len(intersting_layer)):
            grad_cam.backward(idx=idx[0])
            output = grad_cam.generate(target_layer=intersting_layer[i])

            # grad_cam_save('results/{}_grad_cam_{}.png'.format(i, arch), output, raw_img)
            grad_cam_save('results/{}_grad_cam_{}.png'.format(img_name, intersting_layer[i]), output, raw_img)
        del raw_img
    endTime = datetime.datetime.now()
    print("start: "+str(startTime))
    print("end: "+str(endTime))
    # =========================================================================

    # print('Vanilla Backpropagation Visualization')

    # bp = BackPropagation(model=model)
    # probs, idx = bp.forward(image.to(device))

    # for i in range(0, num_class):
    #     bp.backward(idx=idx[i])
    #     output = bp.generate()
    #     grad_save('results/{}_bp_{}.png'.format(CONFIG['target_layer'], arch), output)

    # =========================================================================

    # print('Deconvolution Visualization')

    # deconv = Deconvolution(model=copy.deepcopy(model)) 
    # probs, idx = deconv.forward(image.to(device))

    # for i in range(0, num_class):
    #     deconv.backward(idx=idx[i])
    #     output = deconv.generate()

    #     grad_save('results/{}_deconv_{}.png'.format(CONFIG['target_layer'], arch), output)

    # =========================================================================

    # print('Guided Backpropagation/Guided Grad-CAM Visualization')
   
    # gbp = GuidedBackPropagation(model=model)
    # probs, idx = gbp.forward(image.to(device))

    # for i in range(0, num_class):
    #     grad_cam.backward(idx=idx[i])
    #     region = grad_cam.generate(target_layer=CONFIG['target_layer'])

    #     gbp.backward(idx=idx[i])
    #     feature = gbp.generate()

    #     height, width, _ = feature.shape
    #     region = cv2.resize(region, (width, height))[..., np.newaxis]
    #     output = feature * region

    #     grad_save('results/{}_gbp_{}.png'.format(CONFIG['target_layer'], arch), feature)
    #     grad_save('results/{}_guided_grad_cam_{}.png'.format(CONFIG['target_layer'], arch), output)

if __name__ == '__main__':
    main()
