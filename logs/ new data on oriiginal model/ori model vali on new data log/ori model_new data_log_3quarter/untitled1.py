#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 19:27:13 2023

@author: wyk
"""

import matplotlib.pyplot as plt
myfile1 = open('classification/pointnet2_cls_msg/logs/pointnet2_cls_msg.txt','r')
line1 = myfile1.readlines()
accy = []
count = 1
# Strips the newline character
while count<401:
    
    a = line1[count][-9:-1]      
    accy.append(a)
    count += 4
    
# for item in accy:
#     if item != str:
#         accy.remove(item)



plt.plot(accy)
plt.show()
