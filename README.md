**FOPA: Fast Object Placement Assessment**
=====
This is the PyTorch implementation of **FOPA** for the following research paper.  **FOPA is the first discriminative approach for object placement task.**
> **Fast Object Placement Assessment**  [[arXiv]](https://arxiv.org/pdf/2205.14280.pdf)<br>
>
> Li Niu, Qingyang Liu, Zhenchen Liu, Jiangtong Li

**Our FOPA has been integrated into our image composition toolbox libcom https://github.com/bcmi/libcom. Welcome to visit and try ＼(^▽^)／** 

If you want to change the backbone to transformer, you can refer to [TopNet](https://github.com/bcmi/TopNet-Object-Placement). 

## Setup
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


## Data Preparation
Download and extract data from [Baidu Cloud](https://pan.baidu.com/s/10JBpXBMZybEl5FTqBlq-hQ) (access code: 4zf9) or [Google Drive](https://drive.google.com/file/d/1VBTCO3QT1hqzXre1wdWlndJR97SI650d/view?usp=share_link).
Download the SOPA encoder from [Baidu Cloud](https://pan.baidu.com/s/1hQGm3ryRONRZpNpU66SJZA) (access code: 1x3n) or [Google Drive](https://drive.google.com/file/d/1DMCINPzrBsxXj_9fTKnzB7mQcd8WQi3T/view?usp=sharing). 
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

Download our pretrained model from [Baidu Cloud](https://pan.baidu.com/s/15-OBaYE0CF-nDoJrNcCRaw) (access code: uqvb) or [Google Drive](https://drive.google.com/file/d/1HTP6bSmuMb2Dux3vEX2fJc3apjLBjy0q/view?usp=sharing), and put it in './best_weight.pth'.

## Training
Before training, modify "config.py" according to your need. After that, run:
```
python train.py
```

## Test
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


## Multiple Foreground Scales
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

## Evalution on Discriminative Task

We show the results reported in the paper. FOPA can achieve comparable results with SOPA.
<table>
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>F1</th>
      <th>bAcc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> <a href='https://arxiv.org/pdf/2107.01889.pdf'>SOPA</a> </td>
      <td>0.780</td>
      <td>0.842</td>
    </tr>
    <tr>
      <td>FOPA</td>
      <td>0.776</td>
      <td>0.840</td>
    </tr>
  
  </tbody>
</table>

## Evalution on Generation Task

Given each background-foreground pair in the test set, we predict 16 rationality score maps for 16 foreground scales and generate composite images with top 50 rationality scores. Then, we randomly sample one from 50 generated composite images per background-foreground pair for Acc and FID evaluation, using the test scripts provided by [GracoNet](https://github.com/bcmi/GracoNet-Object-Placement). The generated composite images for evaluation can be downloaded from [Baidu Cloud](https://pan.baidu.com/s/1qqDiXF4tEhizEoI_2BwkrA) (access code: ppft) or [Google Drive](https://drive.google.com/file/d/1yvuoVum_-FMK7lOvrvpx35IdvrV58bTm/view?usp=share_link). The test results of baselines and our method are shown below:

<table>
  <thead>
    <tr style="text-align: right;">
      <th>Method</th>
      <th>Acc</th>
      <th>FID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td> <a href='https://arxiv.org/abs/1904.05475'>TERSE</a> </td>
      <td>0.679</td>
      <td>46.94</td>
    </tr>
    <tr>
      <td><a href='https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123580562.pdf'>PlaceNet</a></td>
      <td>0.683</td>
      <td>36.69</td>
    </tr>
    <tr>
      <td><a href='https://arxiv.org/abs/2207.11464'>GracoNet</a></td>
      <td>0.847</td>
      <td>27.75</td>
    </tr>
    <tr>
      <td><a href='https://openreview.net/pdf?id=hwHBaL7wur'>IOPRE</a></td>  
      <td>0.895</td>
      <td>21.59</td>
    </tr>
    <tr>
      <td>FOPA</td>
      <td> <b>0.932 </td>
      <td> <b>19.76 </td>
    </tr>
  </tbody>
</table>

## Other Resources

+ [Awesome-Object-Placement](https://github.com/bcmi/Awesome-Object-Placement)
+ [Awesome-Image-Composition](https://github.com/bcmi/Awesome-Object-Insertion)


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
