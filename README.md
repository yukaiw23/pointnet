# Pointnet
Pointnet(++) by pytoch in Python

# Data
PS:The training data is already in repository folder, but can also be downloaded from online.
1. Download https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip 'modelnet40'.
2. Save under the folder 'data'. 
3. Download https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip 'shapenet'. 
4. Save under the folder 'data'

# Training
After downloading the dataset, directing the terminal to the working path, then activating the Pytorch environment. 

Run the following for object classification task:

```
python train_classification.py --model pointnet2_cls_msg --normal --log_dir pointnet2_cls_msg --batch_size 16
```

Or run the following for part segmentation task:
```
python train_partseg.py --model pointnet2_part_seg_msg --normal --log_dir pointnet2_part_seg_msg
```
 

The training log and result will be saved under '/log' folder

# Testing
Run the following for object classification taske:
```
python test_classification.py --normal --log_dir pointnet2_cls_msg
``` 


Or run the following for part segentation:
```
python test_partseg.py --normal --log_dir pointnet2_part_seg_msg
``` 


The testing log and result will be saved under '/log' folder

