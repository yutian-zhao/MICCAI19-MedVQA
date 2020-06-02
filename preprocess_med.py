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

lst = []
count = 1
#for cat in ["C1_Modality_train.txt", "C2_Plane_train.txt", "C3_Organ_train.txt", "C4_Abnormality_train.txt"]:
for cat in ["C1_Modality_val.txt", "C2_Plane_val.txt", "C3_Organ_val.txt", "C4_Abnormality_val.txt"]:
#    f = open("ImageClef-2019-VQA-Med-Training/QAPairsByCategory/"+cat, "r", encoding='utf-8')
    f = open("ImageClef-2019-VQA-Med-Validation/QAPairsByCategory/"+cat, "r", encoding='utf-8')
    lines = f.readlines()
    if cat[1] == '1':
        c = 'MODALITY'
    elif cat[1] == '2':
        c = 'PLANE'
    elif cat[1] == "3":
        c = 'ORGAN'
    else:
        c = 'ABN'
    for l in lines:
        d={}
        tok = l.split("|")
    #    tok[2] = tok[2].encode('ascii', 'ignore')
    #    print(tok[2])
        tok[2] = tok[2].replace("\u00a0", "")
        if tok[2].endswith("\n"):
            tok[2] = tok[2][:-1]
        d["qid"] = count
        d["image_name"] = tok[0] + ".jpg"
        d["answer"] = tok[2]
        if tok[2] == "yes" or tok[2] == 'no':
            d["answer_type"] = 'CLOSED'
        else:
            d["answer_type"] = 'OPEN'
        d["question_type"] = c
        d["question"] = tok[1]
        d["phrase_type"] = "freeform"
        lst.append(d)
        count = count + 1
    f.close()
 
#os.makedirs(os.path.dirname("ImageClef-2019-VQA-Med-Training/trainset.json"))
#outfile = "ImageClef-2019-VQA-Med-Training/trainset.json"
outfile = "ImageClef-2019-VQA-Med-Validation/testset.json"
json.dump(lst, open(outfile, 'w'))

#img84_stack = None
#img128_stack = None
#id2idx = {}
#count = 0
#for im_name in os.listdir('data_RAD/images/'):
#    if im_name.endswith(".jpg"):
#        img = cv2.imread('data_RAD/images/' + im_name)
#        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#        img84 = cv2.resize(img_gray, (84, 84))
#        img84 = img84.reshape((1, 84, 84, 1))
#        if img84_stack is None:
#            img84_stack = img84
#        else:
#            img84_stack = np.concatenate((img84_stack, img84), axis=0)
#            
#        img128 = cv2.resize(img_gray, (128, 128))
#        img128 = img128.reshape((1, 128, 128, 1))
#        if img128_stack is None:
#            img128_stack = img128
#        else:
#            img128_stack = np.concatenate((img128_stack, img128), axis=0)
#        
#        id2idx[im_name] = count
#        count = count + 1
#            
#cPickle.dump(img84_stack, open("data_RAD/images84x84.pkl", 'wb'))
#cPickle.dump(img128_stack, open("data_RAD/images128x128.pkl", 'wb'))
#json.dump(id2idx, open("data_RAD/imgid2idx.json", 'w'))
            
            