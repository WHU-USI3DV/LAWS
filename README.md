# Look At the Whole Scene: General Point Cloud Place Recognition by Classification Proxy

## Abtract

Deep learning models centered on retrieval have made significant strides in point cloud place recognition. However, existing approaches struggle to generate discriminative global descriptors and often rely on labor-intensive negative sample mining. Such constraints limit their usability in dynamic and open-world scenarios. To address these challenges, we introduce LAWS, a pioneering classification-centric neural framework that emphasizes looking at the whole scene for superior point cloud descriptor extraction. Central to our approach is the space partitioning design, constructed to provide holistic scene supervision, ensuring the comprehensive learning of scene features. To counteract potential ambiguities arising from the single orthogonal partition boundary, a complementary mechanism of repartitioning space diagonally is specifically designed to dispel classification uncertainties. Under the enhanced partitioning mechanism, the space is separated into several classes and groups. Furthermore, to prevent knowledge forgetting between different groups, a special training strategy is employed, allowing for distinct training of each group. The extensive experiments, encompassing both indoor and outdoor settings and different tasks, validate the generality of LAWS. It not only outperforms contemporary methods but also demonstrates a profound generalization ability across various unseen environments and sensor modalities. Our method achieves a 2.6% higher average top-1 recall on Oxford RobotCar Dataset and a 7.8% higher average recall when generalized to In-house Dataset compared with retrieval-based methods. Furthermore, LAWS also outperforms retrieval-based methods in terms of F1 score, with improvements of 12.7 and 29.2 on the MulRan and KITTI datasets, respectively. Notably, the average localization accuracy of LAWS in indoor environments reached about 68.1%.  Moreover, the scalability and efficiency places LAWS in a leading position for continuous exploration and long-term autonomy. 



## Datasets

-  Oxford RobotCar Dataset
-  In-house Datasets
    - university sector (U.S.)
    - residential area (R.A.)
    - business district (B.D.)
-  MulRan
-  KITTI odometry
-  ScannetPR

Following [PointNetVLAD](https://arxiv.org/abs/1804.03492) the Oxford and In-house datasets can be [downloaded](https://drive.google.com/open?id=1H9Ep76l8KkUpwILY-13owsEMbVCYTmyx) here. MulRan dataset 


## Environment and Dependencies


This project has been tested using Python 3.8.19 with PyTorch 1.10.2 and MinkowskiEngine 0.5.4 on Ubuntu 20.04 with CUDA 11.4. Main dependencies include:

The following packages are required:

- PyTorch (version 1.10.1)
- Torchvision (version 0.11.2)
-  MinkowskiEngine (version 0.5.4)
-  pandas
-  tqdm
-  scipy
-  tensorboardX

Set up the requirments as follows:

1. Create conda environment with python:
  ```
conda create -n LAWS python=3.8
conda activate LAWS
  ```
2. Install PyTorch with suitable cudatoolkit version.

3. Install [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) for MinkLoc3dv2
```
  For example, if you want local MinkowskiEngine
  export CUDA_HOME=/usr/local/cuda-11.X
  git clone https://github.com/NVIDIA/MinkowskiEngine.git
  cd MinkowskiEngine
  python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --blas=openblas
 ```
4. Build the pointops for PPTNet
```
cd libs/pointops && python setup.py install && cd ../../
```

## Training & Evaluating 
1. Train the network
```
sh train.sh
```
2. Evaluate the network
```
sh eval.sh
```


## Acknowledgement

Our code refers to [PointNetVLAD](https://github.com/mikacuy/pointnetvlad), [MinkLoc3Dv2](https://github.com/jac99/MinkLoc3Dv2), [PPT-Net](https://github.com/fpthink/PPT-Net) and [CosPlace](https://github.com/gmberton/CosPlace).


