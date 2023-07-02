# Pointnet
Pointnet(++) by pytoch in Python

### 22/06/2023: 

New committed files needed to be downloaded: 'pointnet2_cls_ssg.py', 'pointnet2_cls_msg.py', 'pointnet2_utils.py', 'residualMLP.py', and 'train_classification.py','ModelNetDaraLoader.py'.

Note:## change the data loader directory in both 'ModelNetDataLoader.py' line 146, 'train_classification.py' line 118.

# Data
PS:The training data is already in repository folder, but can also be downloaded from online.

1. Download https://www.dropbox.com/s/8z2ss05urwqavt6/modelnet40_normal_resampled.zip?dl=0 , The resized/halfed 'modelnet40'.
2. Save under the folder 'data'. 
3. Download https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip 'shapenet'. 
4. Save under the folder 'data'

14/02/2023

Cloud point has been partially cut in half. PLease use the new data folder : "modelnet40_normal_halfsemi" in trainning and testing.

28/02/2023

Cloud point has been updated, with URL : https://www.dropbox.com/s/2dn3tqpip7i7hxw/modelnet40_0228.zip?dl=0, named modenet40_0228

12/04/2023

Cloud point has been segmented into 'half', 'quarter' and 'three quarters' based on its principle axis. The new URL can be found as : https://www.dropbox.com/scl/fo/06oak7qahlup5cupgkjb8/h?dl=0&rlkey=790vt8urzt5debutles496qk8

# Training

### 22/06/2023: 

New committed files needed to be downloaded: 'pointnet2_cls_ssg.py', 'pointnet2_cls_msg.py', 'pointnet2_utils.py', 'residualMLP.py', and 'train_classification.py','ModelNetDaraLoader.py'.

Note:## change the data loader directory in both 'ModelNetDataLoader.py' line 146, 'train_classification.py' line 118.


After downloading the dataset, directing the terminal to the working path, then activating the Pytorch environment. 

### 15/05/2023

Note: ## Besides hyperparameters of Pointnet++, settings for BottleNeck Residual / Inverted Residual MLP needed to be tuned. 

Settings can be done under Class 'BottleNeck', and 'ResidualAdd' in file 'Pointnet2_utils.py'

Run the following for the object classification task:

```
python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg --batch_size 16

python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_cls_ssg --batch_size 16
```

 

The training log and result will be saved under '/log' folder

# Testing
Run the following for object classification task:
```
python test_classification.py --use_normal --log_dir pointnet2_cls_msg
``` 
 


The testing log and result will be saved under '/log' folder

03/05/2023

The model has been trained based on new dataset 'modelnet40_0228', and its test result are: Test Instance Accuracy = 90.3871%, Class accuracy = 88.1154%

The testing demonstration is as below:

 

![截图 2023-03-09 00-35-52](https://user-images.githubusercontent.com/114976583/223773410-cce74421-cd36-46f2-8b33-a85f1d3c8644.png) 



15/05/2023

The data has been updated since 12/04/2023, 

The new testing result is as follows: 
| Data | Train Instance Accuracy | Class Accuracy |
|-----| -------- | -------|
| Half | 94.8943 | 88.1758 | 
| Quarter | 95.4776 | 88.9748 |
| Three Quarters | 95.5589 | 86.7185 |

The new testing result is shown below: 
| Data | Test Instance Accuracy | Class Accuracy |
|-----| -------- | -------|
| Half | 90.3722 | 86.3517 | 
| Quarter | 92.3544 | 88.94 |
| Three Quarters | 88.7136 | 85.4157 |


30/06/2023
With new residual MLP added to the network, the training result is as follows: 
| Data | Train Instance Accuracy | Class Accuracy |
|-----| -------- | -------|
| Half | 95.9756 | 88.5275 | 
| Quarter | 95.5746 | 88.9906 |
| Three Quarters | 95.8943 | 86.4053 |

The testing result is shown below:
| Data | Test Instance Accuracy | Class Accuracy |
|-----| -------- | -------|
| Half |  |  | 
| Quarter |  |  |
| Three Quarters |  |  |

# Data Manipulation
In order to test the robustness of Pointnet++ and the performance of applying it in unexpected and more complex situations, the training data has been modified. One-quarter, half, and three-quarters of the original data are removed with respect to the center of mass of each point cloud

The calculation of the center of mass of point cloud is based on the following equations: 
<p align="center">
   <img src = "https://user-images.githubusercontent.com/114976583/230176228-026453ed-4eb3-4c84-9cf0-6d8794553d97.png"> 
 



   <image src = "https://user-images.githubusercontent.com/114976583/230176365-756d54de-b066-4fd0-b014-4c26fc1ec1bc.png">
</p>

 
where xi, yi, zi are three coordinates of each point within the point cloud, and M, n are a total number of points within the point cloud

Examples of modifying pointcloud of 'airplane' are shown below: 
One quarter / Half /Three-quarters of the point cloud is removed: 
<p align="center">
  <img src = "https://user-images.githubusercontent.com/114976583/230169282-517eed79-8e81-46b1-b23d-2a0a5284b747.png" width="350" height="350"> | <img src = "https://user-images.githubusercontent.com/114976583/230169509-bb729cd7-ee94-4fa9-8564-f245c5ca3c87.png" width="350" height="350">


  <img src = "https://user-images.githubusercontent.com/114976583/230171098-d551fdff-b41e-4a6e-a2a9-cf4cd19571e6.png" width="350" height="350">

</p>

# Network Architecture
Set Abstraction --> BottleNeck Residual / Inverted Residual MLP --> MaxPooling 
