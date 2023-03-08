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

 
Click the following link to interact with pointcloud data.
file:///home/wyk/Point++/Pointnet_Pointnet2_pytorch/data/test.html
![截图 2023-03-09 00-35-52](https://user-images.githubusercontent.com/114976583/223773410-cce74421-cd36-46f2-8b33-a85f1d3c8644.png)
