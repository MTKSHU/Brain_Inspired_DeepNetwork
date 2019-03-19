#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 22:14:28 2019

@author: debo
"""
import cv2
import numpy as np
import os,shutil
#%%

r_array = []
g_array = []
b_array = []
def resizeAndPad(img, size, padColor=0):

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_REPLICATE)
    # scaled_img = cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
    g_array.append(scaled_img[:, :, 1])
    r_array.append(scaled_img[:, :, 0])
    b_array.append(scaled_img[:, :, 2])
    return scaled_img

#%%
path = "/home/ksnehalreddy/Desktop/Fashion Reco/Autoencoder_fashion/"
img_file_names = [x for x in os.listdir(path+"images/") if '.jpg' in x]

#calculate mean and std **********************************************************    
if os.path.exists(path+"resized_images_320_320_padded_white"):
    shutil.rmtree(path+"resized_images_320_320_padded_white")
os.mkdir(path+"resized_images_320_320_padded_white")

# count = 0

with open(path+"error.log","a+") as f:
    for img_f,c in zip(img_file_names,range(len(img_file_names))):
    	# if(count > 10):
    	# 	break
        try:
            img = cv2.imread(path+"images/"+img_f)
            resized_img = resizeAndPad(img, (320, 320), 255)
            cv2.imwrite(path + "resized_images_320_320_padded_white/"+img_f,resized_img)
            print("Done :",c)
            count+=1
        except Exception as e:
            f.write(str(img_f)+","+str(e)+"\n")
        
r_array = np.array(r_array)
g_array = np.array(g_array)
b_array = np.array(b_array)   
r_mean = r_array.mean()
r_std  = r_array.std()
g_mean = g_array.mean()
g_std  = g_array.std()
b_mean = b_array.mean()
b_std  = b_array.std()
 
with open('mean_std.txt','w') as f:
	f.write(str(r_mean) + "," + str(r_std)+"\n")
	f.write(str(g_mean)+ ","+str(g_std)+"\n")
	f.write(str(b_mean)+ ","+str(b_std)+"\n")
