# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:50:45 2020

@author: COMICONO
"""

from __future__ import print_function
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import torch
from language_model import WordEmbedding
from torch.utils.data import Dataset
import itertools
import cv2
import matplotlib.pyplot as plt

#maml_images_data = cPickle.load(open("data_RAD/images128x128.pkl", 'rb'))
#print(type(maml_images_data))
#print(maml_images_data.shape)
#img = maml_images_data[1].reshape((128, 128))
#plt.imshow(img, cmap='gray')
#
#f = open("ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt", "r", encoding='utf-8')
#lines = f.readlines()
#lst = []
#count = 1
#max_length = 0 
#
#for l in lines:
#    d={}
#    tok = l.split("|")
##    tok[2] = tok[2].encode('ascii', 'ignore')
##    print(tok[2])
#    tok[2] = tok[2].replace("\u00a0", "")
#    if tok[2].endswith("\n"):
#        tok[2] = tok[2][:-2]
#    d["qid"] = count
#    d["image_name"] = tok[0] + ".jpg"
#    d["answer"] = tok[2]
#    if tok[2] == "yes" or tok[2] == 'no':
#        d["answer_type"] = 'CLOSED'
#    else:
#        d["answer_type"] = 'OPEN'
#    d["question_type"] = 'UNKNOWN'
#    d["question"] = tok[1]
#    d["phrase_type"] = "para"
#    lst.append(d)
#    count = count + 1
#print(max_length)
#print(sen)   
##os.makedirs(os.path.dirname("ImageClef-2019-VQA-Med-Training/trainset.json"))
#outfile = "ImageClef-2019-VQA-Med-Training/trainset.json"
#json.dump(lst, open(outfile, 'w'))

img84_stack = None
img128_stack = None
id2idx = {}
count = 1
for im_name in os.listdir('data_RAD/images/'):
    if im_name.endswith(".jpg"):
        img = cv2.imread('data_RAD/images/' + im_name)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img84 = cv2.resize(img_gray, (84, 84))
        img84 = img84.reshape((1, 84, 84, 1))
        if img84_stack is None:
            img84_stack = img84
        else:
            img84_stack = np.concatenate(img84_stack, img84, axis=0)
            
        img128 = cv2.resize(img_gray, (128, 128))
        img128 = img84.reshape((1, 128, 128, 1))
        if img128_stack is None:
            img128_stack = img128
        else:
            img84_stack = np.concatenate(img128_stack, img128, axis=0)
        
        id2idx[im_name] = count
        count = count + 1
            
img84_stack.dump_to_file("data_RAD/images84x84.pkl")
img128_stack.dump_to_file("data_RAD/images128x128.pkl")
json.dump(id2idx, open("data_RAD/imgid2idx.json", 'w'))

            
            