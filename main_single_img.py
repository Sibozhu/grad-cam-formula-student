#!/usr/bin/env python
from __future__ import print_function

import copy

import click
import cv2
import numpy as np
import torch
import os, os.path
import datetime
import PIL
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
    flag_data = flag_data.astype(np.float) + raw_img.astype(np.float)
    flag_data = flag_data / flag_data.max() * 255.0
    cv2.imwrite(filename, np.uint8(flag_data))

model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)
model_names.append('yolov3')

model_names.append('yolov3_tiny')

model_names.append('yolov3_medium')


@click.command()
@click.option('-i', '--image-path', help="Path to image", type=str, required=True)
@click.option('-a', '--arch', help="a model name from ```torchvision.models```, e.g., 'resnet152' (default: yolov3)", type=click.Choice(model_names), default='yolov3')
@click.option('-n', '--num_class', help="number of classes to generate (default: 1)", type=int, default=1)
@click.option('--cuda/--no-cuda', help="GPU mode or CPU mode (default: cuda mode)", default=True)
def main(image_path, arch, num_class, cuda):

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
            'input_size': 832
            # 'input_size': 416
        },
        'yolov3_tiny':{
            'target_layer': 'module_list.5',
            'input_size':1536
        },
        'yolov3_medium':{
            'target_layer': 'module_list.5',
            'input_size':832
        }
        # Adding model
    }.get(arch)

    flag_img_name= (image_path.split("/"))
    img_name = flag_img_name[-1]

    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')

    if cuda:
        current_device = torch.cuda.current_device()
        print('Running on the GPU:', torch.cuda.get_device_name(current_device))
    else:
        print('Running on the CPU')

    v3_model_cfg = "./cfg/yolov3_80class_832.cfg"

    #medium yolov3 test 
    v3_medium_cfg = "./cfg/yolov3_medium_test.cfg"

    # tiny_model_cfg = "./cfg/yolov3_tiny_80class_832.cfg"
    tiny_model_cfg = "./cfg/yolov3_tiny_80class_1536.cfg"

    medium_weights_path = "./src/config/medium_832_96.weights"
    
    v3_weights_path = "./src/config/experiments_color_832_baseline_40.weights"
    # tiny_weights_path = "./src/config/vectorized-yolov3-training_january-experiments_tiny_832_98.pt"
    tiny_weights_path = "./src/config/vectorized-yolov3-training_january-experiments_tiny_1536_50.weights"
    if arch == 'yolov3' or arch == 'yolov3_tiny' or arch == 'yolov3_medium':
        if arch == 'yolov3': 
            os.system('./src/utils/darknet/darknet detect ' + v3_model_cfg + ' ' + v3_weights_path + ' ' +image_path + ' -thresh 0.999')
            os.rename("./sample_data/HiveAIRound0_vid_5_frame_2005.jpg", "./results/yolov3_output/" + img_name)

            # os.system('python3 ./src/utils/yolov3/detect.py --image-folder ' + image_path + ' --output-folder ./results/yolov3_output/ --cfg ' + model_cfg + ' --weights ' + weights_path + ' --conf-thres 0.9999 --nms-thres 0.33 --img-size ' + str(CONFIG['input_size']))
            # print("Finished Yolov3 object detection check!")

            with open('./src/config/yolov3_intersting_layers.txt', 'r') as f:
                intersting_layer = f.read().split('\n')
            print('Visualization target layers: '+str(intersting_layer))


        if arch == 'yolov3_medium': 
            os.system('./src/utils/darknet/darknet detect ' + v3_medium_cfg + ' ' + medium_weights_path + ' ' +image_path + ' -thresh 0.999')
            os.rename("./predictions.jpg", "./results/medium_yolo_output/" + img_name)

            # os.system('python3 ./src/utils/yolov3/detect.py --image-folder ' + image_path + ' --output-folder ./results/yolov3_output/ --cfg ' + model_cfg + ' --weights ' + weights_path + ' --conf-thres 0.9999 --nms-thres 0.33 --img-size ' + str(CONFIG['input_size']))
            # print("Finished Yolov3 object detection check!")

            with open('./src/config/yolov3_medium_layers.txt', 'r') as f:
                intersting_layer = f.read().split('\n')
            print('Visualization target layers: '+str(intersting_layer))        



        if arch == 'yolov3_tiny':
            os.system('./src/utils/darknet/darknet detect ' + tiny_model_cfg + ' ' + tiny_weights_path + ' ' +image_path + ' -thresh 0.95 -nms 0.1')
            os.rename("./predictions.jpg", "./results/tiny_yolo_output/" + img_name)

            with open('./src/config/yolov3_tiny_intersting_layers.txt', 'r') as f:
            	intersting_layer = f.read().split('\n')
            print('Visualization target layers: '+str(intersting_layer))

    dictionary = list()
    with open('src/config/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            dictionary.append(line)


    if arch == 'yolov3':
        model = Darknet(v3_model_cfg, CONFIG['input_size'])
        print(arch)
        print('jojojojojojojojojojo')
        model.load_weights(v3_weights_path)

    if arch == 'yolov3_medium':
        model = Darknet(v3_medium_cfg, CONFIG['input_size'])
        print(arch)
        print('jojojojojojojojojojo')
        model.load_weights(medium_weights_path)

    if arch == 'yolov3_tiny':
        model = Darknet(tiny_model_cfg, CONFIG['input_size'])
        if tiny_weights_path.endswith('.weights'):
            print('yoyoyoyoyoyoyoyoyoyo')
            model.load_weights(tiny_weights_path)
        elif tiny_weights_path.endswith('.pt'):
            print('kierankierankierankieran')
            checkpoint = torch.load(tiny_weights_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])

    # else:
    #     model = models.__dict__[arch](pretrained=True)

    #print out all layers' names
    print(*list(model.named_modules()), sep='\n')

    model.to(device)
    model.eval()

    # img_path = image_directory_path
    # ttl_img = []
    # valid_images = [".jpg"]
    # for f in os.listdir(img_path):
    #     ext = os.path.splitext(f)[1]
    #     if ext.lower() not in valid_images:
    #         continue
    #     ttl_img.append((os.path.join(img_path, f)))
    # print(ttl_img)
    # print(str(len(ttl_img)) + " images in total")     



    print('loading '+img_name+' ......')

    if arch == 'yolov3' or arch == 'yolov3_tiny' or arch == 'yolov3_medium':

        raw_img = Image.open(image_path).convert('RGB')
        # w, h = CONFIG['input_size'],CONFIG['input_size']
        w,h = raw_img.width,raw_img.height
        # print(raw_img.size)

        max_dimension = max(h, w)
        pad_w = int((max_dimension - w) / 2)
        pad_h = int((max_dimension - h) / 2)
        # ratio = float(CONFIG['input_size']) / float(max_dimension)

        raw_img = transforms.functional.pad(raw_img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        raw_img = transforms.functional.resize(raw_img, (CONFIG['input_size'], CONFIG['input_size']))
        # raw_img = transforms.functional.resize(raw_img, (w, w))
        flag_image = np.array(raw_img)   
        # print(flag_image.shape)

        raw_img = transforms.functional.to_tensor(raw_img)

        image = raw_img.unsqueeze(0)
        print(image.shape)
        raw_img = raw_img.permute(2, 1, 0)
        # print(raw_img.shape)

        print('loading complete!')
    else:
        raw_image = cv2.imread(image_path)[..., ::-1]
        raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
        image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])(raw_image).unsqueeze(0)

    # =========================================================================

    print('Grad-CAM Visualization on '+img_name)


    grad_cam = GradCAM(model=model)
    probs, idx = grad_cam.forward(image.to(device))

    if arch == 'yolov3' or arch == 'yolov3_tiny' or arch == 'yolov3_medium': 

        for i in range(0,len(intersting_layer)):
            startTime = datetime.datetime.now()

            grad_cam.backward(idx=idx[0])
            output = grad_cam.generate(target_layer=intersting_layer[i])

            # grad_cam_save('results/{}_grad_cam_{}.png'.format(i, arch), output, raw_img)
            grad_cam_save('results/raw/{}_grad_cam_{}.png'.format(img_name, intersting_layer[i]), output, flag_image)

            endTime = datetime.datetime.now()
            time_spent =  endTime - startTime
            grad_cam_image_path = 'results/raw/{}_grad_cam_{}.png'.format(img_name, intersting_layer[i])
            # yolov3_output_path = './results/yolov3_output/' + img_name
            yolov3_output_path = './results/yolov3_output/' + img_name
            medium_output_path = './results/medium_yolo_output/' + img_name
            tiny_output_path = "./results/tiny_yolo_output/" + img_name

            grad_img = Image.open(grad_cam_image_path)
            if arch == 'yolov3':
                pred_img = Image.open(yolov3_output_path).convert('RGB')
            if arch == 'yolov3_medium':
                pred_img = Image.open(medium_output_path).convert('RGB')
            if arch == 'yolov3_tiny':
                pred_img = Image.open(tiny_output_path).convert('RGB')                
            w,h = pred_img.width,pred_img.height
            max_dimension = max(h, w)
            pad_w = int((max_dimension - w) / 2)
            pad_h = int((max_dimension - h) / 2)

            pred_img = transforms.functional.pad(pred_img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
            pred_img = transforms.functional.resize(pred_img, (CONFIG['input_size'], CONFIG['input_size']))

            imgs    = [grad_img, pred_img]
            # # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
            min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
            imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )

            # # save that beautiful picture
            imgs_comb = PIL.Image.fromarray( imgs_comb)
            imgs_comb.save( './results/compare/{}_{}_compare.png'.format(img_name, intersting_layer[i]))  

            print('Time elapsed on layer ['+ intersting_layer[i] + ']: ' + str(time_spent))
        # os.system('open ./results/compare/')
    else:
        for i in range(0, num_class):
            grad_cam.backward(idx=idx[i])
            output = grad_cam.generate(target_layer=CONFIG['target_layer'])

            grad_cam_save('results/open_src_network/{}_gcam_{}.png'.format(dictionary[idx[i]], arch), output, raw_image)
            print('[{:.5f}] {}'.format(probs[i], dictionary[idx[i]]))
    # =========================================================================

    # print('Vanilla Backpropagation Visualization on '+img_name)

    # bp = BackPropagation(model=model)
    # probs, idx = bp.forward(image.to(device))

    # for i in range(0, 1):
    #     bp.backward(idx=idx[0])
    #     output = bp.generate()
    #     grad_save('results/{}_bp.png'.format(img_name), output)

    # =========================================================================

    # print('Deconvolution Visualization on '+img_name)

    # deconv = Deconvolution(model=copy.deepcopy(model)) 
    # probs, idx = deconv.forward(image.to(device))

    # for i in range(0, 1):
    #     deconv.backward(idx=idx[0])
    #     output = deconv.generate()

    #     grad_save('results/{}_deconv.png'.format(img_name), output)

    # =========================================================================

    # print('Guided Backpropagation/Guided Grad-CAM Visualization on '+img_name)
   
    # gbp = GuidedBackPropagation(model=model)
    # probs, idx = gbp.forward(image.to(device))

    # for i in range(0, len(intersting_layer)):
    # 	startTime = datetime.datetime.now()

    # 	grad_cam.backward(idx=idx[0])
    # 	region = grad_cam.generate(target_layer=intersting_layer[i])

    # 	gbp.backward(idx=idx[0])
    # 	feature = gbp.generate()

    # 	height, width, _ = feature.shape
    # 	region = cv2.resize(region, (width, height))[..., np.newaxis]
    # 	output = feature * region

    #     grad_save('results/{}_gbp_{}.png'.format(img_name, intersting_layer[i]), feature)
    # 	grad_save('results/{}_guided_grad_cam_{}.png'.format(img_name, intersting_layer[i]), output)
    # 	endTime = datetime.datetime.now()
    # 	time_spent =  endTime - startTimeprint('Time spent: '+ str(time_spent))

if __name__ == '__main__':
    main()

