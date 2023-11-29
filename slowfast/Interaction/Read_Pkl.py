#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 14:34:43 2021

@author: anslab
"""
#### load results.pkl
import pickle
import torch

results_data=[];
results_file_path='/home/tsaim/anaconda3/envs/pyslowfast/slowfast/slowfast/Interaction_test/output/Prediction_GT_LOSOCV_HomeLab_subj1_manipulation_wo_Reverse_sqe_32_stride_16_wo_Reverse_wo_flip.pickle';
with open(results_file_path, 'rb') as f:
    results_data.append(pickle.load(f))
    
print(results_data)
