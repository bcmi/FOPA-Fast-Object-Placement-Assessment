**FOPA: Fast Object Placement Assessment**
=====
This is the PyTorch implementation of **FOPA** for the following research paper:
> **Fast Object Placement Assessment**  [[arXiv]](https://arxiv.org/pdf/2205.14280.pdf)<br>
>
> Li Niu, Qingyang Liu, Zhenchen Liu, Jiangtong Li


# Setup
All the code have been tested on PyTorch 1.7.0. Follow the instructions to run the project.

First, clone the repository:
```
git clone git@github.com:bcmi/FOPA-Fast-Object-Placement-Assessment.git
```
Then, install Anaconda and create a virtual environment:
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
Download and extract data from [Baidu Cloud](https://pan.baidu.com/s/10JBpXBMZybEl5FTqBlq-hQ) (access code: 4zf9).
Download the SOPA encoder from [Baidu Cloud](https://pan.baidu.com/s/1hQGm3ryRONRZpNpU66SJZA) (access code: 1x3n). 
Put them in "data/data". It should contain the following directories and files:
```
<data/data>
  bg/                         # background images
  fg/                         # foreground images
  mask/                       # foreground masks
  train(test)_pair_new.json   # json annotations 
  train(test)_pair_new.csv    # csv files
  SOPA.pth.tar                # SOPA encoder
```

Download our pretrained model from [Baidu Cloud](https://pan.baidu.com/s/15-OBaYE0CF-nDoJrNcCRaw) (access code: uqvb), and put it in './best_weight.pth'.

# Traing
Before training, modify "config.py" according to your need. After that, run:
```
python train.py
```

# Test
To get the F1 score and balanced accuracy of a specified model, run:
```
python test.py --mode evaluate 
```

The results obtained with our released model should be F1: 0.778302, bAcc: 0.838696.


To get the heatmaps predicted by FOPA, run:
```
python test.py --mode heatmap
```

To get the optimal composite images based on the predicted heatmaps, run:
```
python test.py --mode composite
```


# Multiple Foreground Scales
For testing multi-scale foregrounds for each foreground-background pair, first run the following command to generate 'test_data_16scales.json' in './data/data' and 'test_16scales' in './data/data/fg', './data/data/mask'.
```
python prepare_multi_fg_scales.py
```

Then, to get the heatmaps of multi-scale foregrounds for each foreground-background pair, run:
```
python test_multi_fg_scales.py --mode heatmap
```

Finally, to get the composite images with top scores for each foreground-background pair, run:
```
python test_multi_fg_scales.py --mode composite
```

# Other Resources

+ [Awesome-Object-Placement](https://github.com/bcmi/Awesome-Object-Placement)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Image-Composition)


## Bibtex

If you find this work useful for your research, please cite our paper using the following BibTeX  [[arxiv](https://arxiv.org/pdf/2107.01889.pdf)]:

```
@article{niu2022fast,
  title={Fast Object Placement Assessment},
  author={Niu, Li and Liu, Qingyang and Liu, Zhenchen and Li, Jiangtong},
  journal={arXiv preprint arXiv:2205.14280},
  year={2022}
}
```
