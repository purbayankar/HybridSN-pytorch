# Hybrid-Spectral-Net for Hyperspectral Image Classification.

## Description

The  HybridSN  is  spectral-spatial  3D-CNN  followed  by spatial 2D-CNN. The 3D-CNN facilitates the joint spatial-spectral feature  representation  from  a  stack  of  spectral  bands.  The  2D-CNN  on  top  of  the  3D-CNN  further  learns  more  abstract  level spatial  representation. 

## Model

<img src="HSI-RN.jpg"/>

Fig: HybridSpectralNet (HybridSN) Model with 3D and 2D convolutions for hyperspectral image (HSI) classification.

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
```

To download the dataset and setup the folders, run:

```
bash setup_script.sh
```

## Training

To train the model(s) in the paper, run this command in the A2S2KResNet folder:

```train
python A2S2KResNet.py -d <IN|UP|KSC> -e 200 -i 3 -p 3 -vs 0.9 -o adam
```
