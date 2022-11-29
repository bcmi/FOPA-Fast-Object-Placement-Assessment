**FOPA: Fast Object Placement Assessment**
=====
This is the PyTorch implementation of **FOPA**, the paper can be found in [here](https://arxiv.org/pdf/2205.14280.pdf). 

# Setup
All the code have been tested on PyTorch 1.7.0, following the instructions to run the project.

First, clone the repository:
```
git clone git@github.com:bcmi/FOPA-Fast-Object-Placement-Assessment.git
```
Then, install Anaconda and create a vitrual environment:
```
conda create -n fopa
conda activate fopa
```
Install PyTorch 1.7.0 (higher version should be fine):
```
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=10.2 -c pytorch
```
Install necessary packages:
```
pip install -r requirements.txt
```
# Data Preparation
Download and extrace data from [Baidu Cloud](https://pan.baidu.com/s/18FfLt7NCuL4BRhpsDIikBA?pwd=spk7) (access code: spk7); 
Download the SOPA encoder from [Baidu Cloud](https://pan.baidu.com/s/1wHMyVq6XjjxzZZC9JXwLJQ ) (access code: 5gug)
Put them in "data/data". It should contain following directories and files:
```
<data/data>
  bg/                         # background images
  fg/                         # foreground images
  mask/                       # mask images for the foreground
  train2017/                  # COCO dataset 2017, training set
  depth_map/                  # depth of the background 
  semantic_newlabel/          # stuff and thing annotation for the backgound
  train(test)_pair_new.json   # json annotations 
  train(test)_pair_new.json,  # csv files
  Net_best.pth.tar            # SOPA encoder(resnet 18)
```

Download the best model from [Baidu Cloud](https://pan.baidu.com/s/13ubY2luNuoMf3VdqGurxKw) (access code: ui0c), and put it in './best_weight.pth'.
# Traing
Before training, modify the trainging parameters in "config.py" accroding to your need, such as "ex_name", "gpu_id" and "batch_size". 
After that, run:
```
python train.py
```

# test
To get the F1 score and balanced accuracy of a specified model, run:
```
python test.py 
```
Or specify the path to the model (the default path is './best_weight.pth'), run:
```
python test.py --path <model path>
```

To get the heatmaps of train or test set, run:
```
python test.py --mode train/test
```
You can get all the heatmaps of the best model from [Baidu Cloud](https://pan.baidu.com/s/1snK2PKZtwnDG56wL4BWuMA) (access code: 29w6)
