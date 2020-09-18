#!/usr/bin/python3

import argparse
import os
import random
import tempfile
import time

from PIL import Image, ImageDraw
import torch

import torchvision
from models import Darknet
from utils.datasets import load_images_and_labels
from utils.nms import nms
from utils.utils import xywh2xyxy
from utils import storage_client

def main(image_uri,
         output_uri,
         weights_uri,
         model_cfg,
         img_size,
         bw,
         conf_thres,
         nms_thres):

    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

    model = Darknet(model_cfg, img_size)

    # Load weights
    weights_path = storage_client.get_file(weights_uri)
    if weights_path.endswith('.weights'):  # darknet format
        model.load_weights(weights_path)
    elif weights_path.endswith('.pt'):  # pytorch format
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    model.to(device, non_blocking=True)

    detect(image_uri,
           output_uri,
           model,
           img_size,
           bw,
           device=device,
           conf_thres=conf_thres,
           nms_thres=nms_thres)


def detect(image_uri,
           output_uri,
           model,
           img_size,
           bw,
           device,
           conf_thres=0.5,
           nms_thres=0.25):

        if image_uri.startswith('gs'):
            img_filepath = storage_client.get_uri_filepath(image_uri)
        else:
            img_filepath = image_uri
        img = Image.open(img_filepath).convert('RGB')
        w, h = img.size

        max_dimension = max(h, w)
        pad_w = int((max_dimension - w) / 2)
        pad_h = int((max_dimension - h) / 2)
        ratio = float(img_size) / float(max_dimension)

        img = torchvision.transforms.functional.pad(img, padding=(pad_w, pad_h, pad_w, pad_h), fill=(127, 127, 127), padding_mode="constant")
        img = torchvision.transforms.functional.resize(img, (img_size, img_size))

        if bw:
            img = torchvision.transforms.functional.to_grayscale(img, num_output_channels=1)

        img = torchvision.transforms.functional.to_tensor(img)
        img = img.unsqueeze(0)

        with torch.no_grad():
            model.eval()
            img = img.to(device, non_blocking=True)
            output = model(img)
            print("OUTPUT SHAPE", output.shape)

            for detections in output:
                detections = detections[detections[:, 4] > conf_thres]
                # From (center x, center y, width, height) to (x1, y1, x2, y2)
                box_corner = torch.zeros((detections.shape[0], 4), device=detections.device)
                xy = detections[:, 0:2]
                wh = detections[:, 2:4] / 2
                box_corner[:, 0:2] = xy - wh
                box_corner[:, 2:4] = xy + wh
                probabilities = detections[:, 4]
                nms_indices = nms(box_corner, probabilities, nms_thres)
                box_corner = box_corner[nms_indices]
                if nms_indices.shape[0] == 0:  
                    continue

                img_with_boxes = Image.open(img_filepath)
                draw = ImageDraw.Draw(img_with_boxes)
                w, h = img_with_boxes.size

                for i in range(len(box_corner)):
                    x0 = box_corner[i, 0].to('cpu').item() / ratio - pad_w
                    y0 = box_corner[i, 1].to('cpu').item() / ratio - pad_h
                    x1 = box_corner[i, 2].to('cpu').item() / ratio - pad_w
                    y1 = box_corner[i, 3].to('cpu').item() / ratio - pad_h
                    draw.rectangle((x0, y0, x1, y1), outline="red")

                with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
                    img_with_boxes.save(temp_file.name)
                    storage_client.upload_file(temp_file.name, output_uri)

if __name__ == '__main__':
    GS_PREFIX="gs://mit-driverless/vectorized-yolov3-training"
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_cfg', type=str, default='model_cfgs/yolov3.cfg')
    parser.add_argument('--image_uri', type=str,
                        default="gs://image1.jpg gs://image2.jpg")
    parser.add_argument('--output_uri', type=str,
                        default="/test.csv")
    parser.add_argument('--bw', help='Enable black and white', action='store_true')
    parser.add_argument('--weights_uri', type=str,
                        default=os.path.join(GS_PREFIX, "sample-yolov3.weights"))
    parser.add_argument('--conf_thres', type=float, default=0.999,
                        help='object confidence threshold')
    parser.add_argument('--nms_thres', type=float, default=0.1,
                        help='IoU threshold for non-maximum suppression')
    parser.add_argument('--iou_thres', type=float, default=0.25,
                        help='IoU threshold required to qualify as detected')
    parser.add_argument('--img_size', type=int, default=416)
    opt = parser.parse_args()
    print("Arguments:", opt)

    main(image_uri=opt.image_uri,
         output_uri=opt.output_uri,
         weights_uri=opt.weights_uri,
         model_cfg=opt.model_cfg,
         img_size=opt.img_size,
         bw=opt.bw,
         conf_thres=opt.conf_thres,
         nms_thres=opt.nms_thres)
