
## Exploiting application of missing point clouds on the classification using PointNet-like architecture

This repo works as a guide and record of the above topic. 

### 05/07/2023: 

New committed files needed to be downloaded: 'pointnet2_cls_ssg.py', 'pointnet2_cls_msg.py', 'pointnet2_utils.py', 'residualMLP.py', and 'train_classification.py','ModelNetDaraLoader.py'.
In 'train_classification.py', make sure line 129, 130, 131 have the same name as the new files you downloaded and saved above. 

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
python test_classification.py --log_dir pointnet2_cls_msg
``` 
 


The testing log and result will be saved under '/log' folder

05/03/2023

The ModelNet40 has been removed quarter, half and three quarters of it, with respect to offset world coordinate of points cloud.

The model has been trained based on new dataset 'modelnet40_0228', and its test result are: Test Instance Accuracy = 90.3871%, Class accuracy = 88.1154%

The testing demonstration is as below:

<div align=center>
<img src="https://user-images.githubusercontent.com/114976583/223773410-cce74421-cd36-46f2-8b33-a85f1d3c8644.png" width="800" height="400">
</div>

However, the method we used to manipulate datasets is based on the default world coordinate of the ModelNet40, which cause uneven removal. For example, sometimes we want to remove half of the points cloud, but more than / fewer portion is removed. Then we decided to pursue the way introduced below  'Data Manipulation'

15/05/2023

The data has been updated since 12/04/2023, as the new manipulated dataset is used to train the original PoinNet++ model.

The new testing result is as follows: 
| Data | Train Instance Accuracy | Test Accuracy |
|-----| -------- | -------|
| Half | 94.8943 | 88.1758 | 
| Quarter | 95.4776 | 88.9748 |
| Three Quarters | 95.5589 | 86.7185 |

The validation result is shown below: 
| Data | Validation Accuracy| 
|-----| -------- |
| Half | 86.7 | 
| Quarter | 88.97 | 
| Three Quarters | 86.39 | 


30/06/2023
With new residual MLP added to the network, the training result is as follows: 
| Data | Train Instance Accuracy | Test Accuracy |
|-----| -------- | -------|
| Half | 92.937 | 89.0726 | 
| Quarter | 93.171 | 90.5242 |
| Three Quarters | 92.2866 | 87.2177 |

The validation result is shown below:
| Data | Validation Accuracy | 
|-----| -------- |
| Half | 87.5585  |  
| Quarter | 89.1393 | 
| Three Quarters | 85.4411 |

# Data Manipulation
In order to test the robustness of Pointnet++ and the performance of applying it in unexpected and more complex situations, the training data has been modified. One-quarter, half, and three-quarters of the original data are removed with respect to the center of mass of each point cloud

The calculation of the center of mass of point cloud is based on the following equations: 
<p align="center">
 
   <img src = "https://user-images.githubusercontent.com/114976583/230176228-026453ed-4eb3-4c84-9cf0-6d8794553d97.png">
 
</p>
<p align="center">

   <image src = "https://user-images.githubusercontent.com/114976583/230176365-756d54de-b066-4fd0-b014-4c26fc1ec1bc.png">

</p>

 
where xi, yi, zi are three coordinates of each point within the point cloud, and M, n are a total number of points within the point cloud

Examples of a modifying point cloud of 'airplane' are shown below: 
One-quarter / Half /Three-quarters of the point cloud is removed: 

<div align=center>
 
![image](https://github.com/yukaiw23/pointnet/assets/114976583/7c9f4b08-79eb-4686-8313-d7b5d203f375)


</div>

# Network Architecture
Set Abstraction --> BottleNeck Residual / Inverted Residual MLP --> MaxPooling 

Traditional Residual Blk:
<div align=center>
<img src="https://github.com/yukaiw23/pointnet/assets/114976583/710fc015-785a-4a69-bf76-4234f0625da6.jpg" width="400" height="300">
</div>

Our Residual Blk that was inserted into PointNet++:

<div align=center>
 
![image](https://github.com/yukaiw23/pointnet/assets/114976583/6d306fd9-bf93-4188-b97d-3eaa509c05b4)

</div>

# Results
Our model is trained with manipulated datasets from ModelNet40 (quarter, half, and three-quarters removed)

The training accuracy is shown below: 

<div align=center>

![image](https://github.com/yukaiw23/pointnet/assets/114976583/5c341b45-8b6d-4271-a4a6-aa959626019a|width=100)

</div>

The testing accuracy is as follows:

<div align=center>
 
![image](https://github.com/yukaiw23/pointnet/assets/114976583/bb85b166-95f3-43b1-99ce-8d221dc54568)

</div>


Validation accuracy is below:

<div align=center>
 
<img src="https://github.com/yukaiw23/pointnet/assets/114976583/1dc88bd3-a4de-40bd-b67f-c12dd99a237e.jpg" width="400" height="300">
</div>

With a simple MLP module inserted, our model only uses around 0.8 million more parameters than the original PointNet++, and 1.2 million parameters less than PointNet

The validation accuracy of new datasets on original PoineNet++ and retrained PointNet++ is shown:

<div align=center>

<img src="https://github.com/yukaiw23/pointnet/assets/114976583/b9301abf-e3c6-49e5-8384-7fbb44541377.jpg" width="400" height="300"> 

</div>



