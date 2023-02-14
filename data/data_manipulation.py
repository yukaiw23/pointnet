#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 01:22:26 2023

@author: wyk
"""
import os


directory = 'modelnet40_normal_resampled/xbox/' #where the original files are



for filename in os.listdir(directory): # list all files in directory
    
    my_file = open('modelnet40_normal_resampled/xbox/'+filename,'r') # open and read all files in directory

    data = my_file.read()
    data = list(data.split("\n"))
    data = [item.split(',') for item in data]


    file = open('modelnet40_normal_halfsemi/xbox/'+filename,'w') # create new file in new location, and process the following.

    for i in range(len(data)-1):
    
        if float(data[i][0]) > 0:
        
        
            file.writelines(data[i][0]+","+data[i][1]+","+data[i][2]+","+data[i][3]+","+ data[i][4]+","+data[i][5])
            file.write("\n")
        
        elif float(data[i][0]) == 0:
           
        
            file.writelines(data[i][0]+","+data[i][1]+","+data[i][2]+","+data[i][3]+","+ data[i][4]+","+data[i][5])
            file.write("\n")


    file.close() # save the processed file in custoermized location
    
        
    
    
