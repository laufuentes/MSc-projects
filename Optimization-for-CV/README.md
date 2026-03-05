# Ultrasound speckle reduction using adaptive wavelet thresholding
#### Optimization for Computer Vision Project

Laura Fuentes Vicente, M2 Mathematics and Artificial Intelligence, Paris-Saclay University 2023/2024

### Introduction 
Welcome to my Github repository for the project of Optimization for Computer Vision. For this project we aimed to analyze an article focused on Optimization and computer vision, construct a report and implement some of the algorithms presented. I chose to work on the paper: *Ultrasound speckle reduction using adaptive wavelet thresholding* presented by Anterpreet Kaur Bedi and Ramesh Kumar Sunkaria in 2022 [1]. 

Medical ultrasonography serves as a crucial imaging modality in clinical diagnosis, relying on the principle of acoustic impedance. The efficiency of medical ultrasonography can be impacted by various constraints, including speckle, acoustic shadows, artifacts, etc. This study specifically addresses the issue of speckle, a granular multiplicative noise that degrades texture information and obscures details such as lines, edges, and boundaries in an image. The authors introduce, more precisely, a novel multi-scale and adaptive technique for ultrasound image despeckle. The technique proposed features a unique thresholding function, which progressively reduces wavelet coefficients to zero when their values fall below a specified threshold. This approach is based on two key principles: firstly, the statistical properties of the image across various decomposition levels, and secondly, the assumption that speckle predominantly manifests in low-valued wavelet coefficients.

### Guide 
To run the implementations and recreate the report images, I created four guided python notebooks. All functions deployed in notebooks are gathered in the folder \src. 
In the \image folder you may find the resulting images and datasets from notebooks. Finally, in the folder \Dataset_BUSI_with_GT, we can find a reduced version of the ultrasound image dataset from Kaggle [2]. Our dataset contains 2 normal ultrasound image, 1 benign and 1 malign. 

In order to run properly all notebooks, we propose to run the following command to download the required python packages for implementation: 


With pip: 

```
pip install numpy scipy matplotlib seaborn opencv-python PyWavelets scikit-learn
```


With conda: 


```
conda install numpy
conda install -c conda-forge opencv
conda install PyWavelets
conda install pandas
conda install seaborn
conda install scikit-learn
conda install -c anaconda scipy
conda install -c conda-forge matplotlib
```

[1] https://idp.springer.com/authorize/casa?redirect_uri=https://link.springer.com/article/10.1007/s11045-021-00799-4&casa_token=HySOHIRizsMAAAAA:tXK0_ObkRiIlNKJ8m3Zz8e_kWEgiza0RjYKOya5IBrW723lsM3liut2xXt5HI8cEeK7IVeuGSJ98TxSy 


[2] https://www.kaggle.com/datasets/anaselmasry/datasetbusiwithgt
