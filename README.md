# Sample Augmentation Coupled with Sample Migration for Time-series Remote Sensing Data Land Cover mapping
This is a official implementation for our paper: "Sample Augmentation Coupled with Sample Migration for Time-series Remote Sensing Data Land Cover mapping" by Zhiyong Lv, Ziqing Zhao, Pengfei Zhang, Linfu Xie, Jón Atli Benediktsson, Zhenong Jin, and Rui Zhu.  

This repository provides the official implementation of a novel approach for Time-Series Land Cover Mapping (T-LCM). Traditional T-LCM methods are hindered by the massive cost of manually labeling samples for each image in a sequence.
Our method dramatically reduces this burden by requiring only a single, small set of manually labeled samples for one reference image to generate accurate maps for an entire time series of remote sensing images.

![The flowchart of the proposed approach for achieving T-LCM with SRSIs.](https://github.com/ImgSciGroup/T-LCM/blob/main/Figures/Figure1.png)

## Requirements
>python=3.7.10  
pytorch=1.9  
opencv-python=4.1.0.25  
scikit-image=0.14.2  
scikit-learn=0.24.1  

## Dataset
Six study sites in a global range, including Wuhan, China (S1), Shenzhen, China (S2), Melbourne, Australia (S3), Berlin, Germany (S4), Cairo, Egypt (S5), and San Francisco (S6),which are collected to verify the feasibility and robustness of the proposed approach.
![Six study sites are selected in a worldwide range to verify the performance of the proposed approach. The green points illustrate the
location of the six study sites, and the table at the bottom of the figure summarizes the details of the referred SRSIs for the six study sites.](https://github.com/ImgSciGroup/T-LCM/blob/main/Figures/Figure2.png)
The datasets for sites S1 to S5 are available at [TSSCD](https://github.com/CUG-BEODL/TSSCD), and the dataset for S6 can be downloaded from https://pan.baidu.com/s/1P_7yFr9UfoOY0dh-0ilwfg?pwd=tkij.

## Acknowledgement
We are very grateful for the outstanding contributions of the publicly available datasets (S1 to S5) of the papers [1].  
```
[1] He H, Yan J, Liang D, et al. Time-series land cover change detection using deep learning-based temporal semantic segmentation[J]. Remote Sensing of Environment, 2024, 305: 114101.
```

## Contact us 
If you have any problme when running the code, please do not hesitate to contact us. Thanks.  
E-mail: Lvzhiyong_fly@hotmail.com
