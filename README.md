# GRAD-CAM on YoloV3 

<p align="center">
<img src="https://user-images.githubusercontent.com/22118253/93553466-0c228200-f941-11ea-9c99-36627ab0d2c6.png" width="800">
</p>

## 1. check dependencies.

See file `dependencies.txt`. To install: `pip install -r dependencies.txt`.

* python 2.7
* torch 0.4.1
* torchvision 0.2.1
* opencv-python 3.4.3.18
* click 6.7
* numpy 1.15.4
* pillow 5.3.0


# Usage

## Visualization
### Single Image Visualization
e.g: If you wish to do visualization of "HiveAIRound1_vid_18_frame_1068.jpg" on our YoloV3 model with only 1 class and GPU mode off, type:
```
python3 main_single_img.py -i "./sample_data/HiveAIRound2_vid_37_frame_404.jpg" --no-cuda -a yolov3
```

### YoloV3 tiny sample
```
python3 main_single_img.py -i "./sample_data/HiveAIRound2_vid_37_frame_404.jpg" --no-cuda -a yolov3_tiny
```

e.g: If you wish to do visualization of "cat_dog.png" on resnet152 model with 3 classes and GPU mode off, type:
```
python3 main_single_img.py -i "./sample_data/cat_dog.png" -a resnet152 -n 3 --no-cuda
```

### Folder Visualization
e.g: If you wish to do visualization of one whole folder's(for example "./data/HiveAIRound0/") all .jpg images, with GPU mode on, type:
```
python3 main_folder.py -i "./data/HiveAIRound0/" --cuda
```

### Mass Visualization

In order to solve the GPU/CPU out of memory error when applying for loop inside python script, I wrote a BATCH script "mass_main" that loops all images in dataset and do single image visualization one at time. Fortunately this works pretty well! Windows user can just 
execute:
```
mass_main.bat
```