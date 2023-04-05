# Pointnet
Pointnet(++) by pytoch in Python

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
# Training

After downloading the dataset, directing the terminal to the working path, then activating the Pytorch environment. 

Run the following for object classification task:

```
python train_classification.py --model pointnet2_cls_msg --log_dir pointnet2_cls_msg --batch_size 16
```

Or run the following for part segmentation task:
```
python train_partseg.py --model pointnet2_part_seg_msg --log_dir pointnet2_part_seg_msg
```
 

The training log and result will be saved under '/log' folder

# Testing
Run the following for object classification taske:
```
python test_classification.py --use_normal --log_dir pointnet2_cls_msg
``` 


Or run the following for part segentation:
```
python test_partseg.py --use_normal --log_dir pointnet2_part_seg_msg
``` 


The testing log and result will be saved under '/log' folder

05/03/2023

The model has been trained based on new dataset 'modelnet40_0228', and its test result are: Test Instance Accuracy = 90.3871%, Class accuracy = 88.1154%

The testing demonstration is as below:

 

![截图 2023-03-09 00-35-52](https://user-images.githubusercontent.com/114976583/223773410-cce74421-cd36-46f2-8b33-a85f1d3c8644.png)


# Data Manipulation
In order to test robustness of Pointnet++ and the performance of applying it in unexpected and more complex situation, the training data has been modified. One quarter, half and three quarters of the original data are removed with respect to center of mass of each pointcloud

Examples of modifying pointcloud of 'airplane' are shown below: 
1. One quarter of the pointcloud is removed: 

![image](https://user-images.githubusercontent.com/114976583/230169282-517eed79-8e81-46b1-b23d-2a0a5284b747.png =250x250)

2. Half of the pointcloud is removed:

![image](https://user-images.githubusercontent.com/114976583/230169509-bb729cd7-ee94-4fa9-8564-f245c5ca3c87.png)

3. Three quarters of the pointcloud is removed:

![image](https://user-images.githubusercontent.com/114976583/230169709-79909c8e-72d8-4de8-93ce-73822b012124.png)


