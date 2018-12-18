#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Image Preprocessed


# In[1]:


import os
import glob
import shutil
import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from tqdm import tqdm
from PIL import Image
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from skimage.feature import hog
from skimage.feature import haar_like_feature
from skimage.feature import local_binary_pattern as lbp
from skimage.io import imread
from skimage.transform import pyramid_gaussian, integral_image
from sklearn.externals import joblib

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def resize_by_short(img, short_len=64):
    """按照短边进行所需比例缩放"""
    (x, y) = img.size
    if x>y:
        y_s=short_len
        x_s=int(x*y_s/y)
        x_l=int(x_s/2)-int(short_len/2)
        x_r=int(x_s/2)+int(short_len/2)
        img = img.resize((x_s, y_s))
        box = (x_l, 0, x_r, short_len)
        img = img.crop(box)
    else:
        x_s=short_len
        y_s=int(y*x_s/x)
        y_l=int(y_s/2)-int(short_len/2)
        y_r=int(y_s/2)+int(short_len/2)
        img = img.resize((x_s, y_s))
        box = (0, y_l, short_len, y_r)
        img = img.crop(box)
    return img


# In[3]:


# feature extraction


# In[4]:


def image_preprocess_size(args=object, short_len = 64):
    pPath = args.DIR_PATHS["POS_IMG_PH"] + '/'
    nPath = args.DIR_PATHS["NEG_IMG_PH"] + '/'
    pImages=[pPath+x for x in os.listdir(pPath) if not x.startswith('.')]
    nImages=[nPath+x for x in os.listdir(nPath) if not x.startswith('.')]
    
    sizes_pos = []
    sizes_neg=[]
    for img_name in pImages:
        img = Image.open(img_name)
        sizes_pos.append(img.size)
        img=resize_by_short(img, short_len)
        img.save(pPath + os.path.split(img_name)[1])

    for img_name in nImages:
        img = Image.open(img_name)
        sizes_neg.append(img.size)
        img=resize_by_short(img, short_len)
        img.save(nPath + os.path.split(img_name)[1])


# In[5]:


def process_image(image, args=object):
    if args.DES_TYPE == "HOG":
        fd = hog(image, block_norm='L2', pixels_per_cell=args.PIXELS_PER_CELL)
    elif args.DES_TYPE == "LBP":
        fd = lbp(image, args.LBP_POINTS, args.LBP_RADIUS)
    elif args.DES_TYPE == "HAAR":
        fd = haar_like_feature(integral_image(image), 0, 0, 5, 5, 'type-3-x')
    else:
        raise KeyError("==> The Processing method does not exist!")
    return fd


# In[6]:


def extract_features(args=object):
    if os.path.exists(args.DIR_PATHS["POS_FEAT_PH"]):
        shutil.rmtree(args.DIR_PATHS["POS_FEAT_PH"])
    if os.path.exists(args.DIR_PATHS["NEG_FEAT_PH"]):
        shutil.rmtree(args.DIR_PATHS["NEG_FEAT_PH"])
    os.makedirs(args.DIR_PATHS["POS_FEAT_PH"])
    os.makedirs(args.DIR_PATHS["NEG_FEAT_PH"])

    print("==> Calculating the descriptors for the positive samples and saving them")
    for im_path in tqdm(glob.glob(os.path.join(args.DIR_PATHS["POS_IMG_PH"], "*"))):
        im = imread(im_path, as_grey=True)
        fd = process_image(im,args)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(args.DIR_PATHS["POS_FEAT_PH"] ,fd_name)
        joblib.dump(fd, fd_path)
    print("==> Positive features saved in {}".format(args.DIR_PATHS["POS_FEAT_PH"]))

    print("==> Calculating the descriptors for the negative samples and saving them")
    for im_path in tqdm(glob.glob(os.path.join(args.DIR_PATHS["NEG_IMG_PH"], "*"))):
        im = imread(im_path, as_grey=True)
        fd = process_image(im,args)
        fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
        fd_path = os.path.join(args.DIR_PATHS["NEG_FEAT_PH"], fd_name)
        joblib.dump(fd, fd_path)
    print("==> Negative features saved in {}".format(args.DIR_PATHS["NEG_FEAT_PH"]))
    print("==> Completed calculating features from training images")


# In[ ]:




