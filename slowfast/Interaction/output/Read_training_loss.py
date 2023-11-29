#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 12:05:28 2021

@author: tsaim
"""

import numpy as np
import matplotlib.pyplot as plt

log_filepath='/home/tsaim/anaconda3/envs/pyslowfast/slowfast/slowfast/Interaction/output/stdout.log';
with open(log_filepath, 'r') as b:               
    lines = [line.strip() for line in b]

# save_log_filepath='/home/tsaim/anaconda3/envs/pyslowfast/slowfast/slowfast/Interaction/output/training_log_TrainedOnHomeLab.txt'
# #save training log
# output_file = open(save_log_filepath, 'w')

train_epoch=[];train_loss=[];train_lr=[];train_top1_err=[];train_top5_err=[];
val_epoch=[];val_loss=[];val_top1_err=[];val_top5_err=[];loaded_model=[];
for i in range(0,len(lines)):#174380
    log_note=lines[i];
    if 'CHECKPOINT_FILE_PATH' in log_note:#loaded weights
        loaded_model.append(log_note);
    if '_type' in log_note:
        _type=log_note.split('_type": "')[1].split('"')[0];
        # Interaction: Trained on HomeLab only model ran between Sep.1-15, 2021
        # Interaction: Trained on HomeLab and Home model ran between Nov.1-21, 2021
        # Stabilization: Trained on HomeLab model ran between Oct.5-9(10:42), 2021 and between Oct.29-30, 2021
        # Manipulation: Trained on HomeLab model ran between Oct.9(13:30)-Oct.18, 2021
        
        month=log_note.split('/')[0].split('[')[1].split(']')[0];
        day=log_note.split('/')[1].split(' ')[0];
        hrs=log_note.split('[')[1].split(']')[0].split(':')[0].split(' ')[1];
        mins=log_note.split('[')[1].split(']')[0].split(':')[1];
        secs=log_note.split('[')[1].split(']')[0].split(':')[2];
        if (int(month)<13) and (int(day)<32):#in training             
            #save training & validation process  
            # output_file.write(log_note+'\n');            
            epoch=int(log_note.split('epoch": "')[1].split('"')[0].split('/')[0]);
            top1_err=np.float(log_note.split('"top1_err": ')[1].split(',')[0]);
            if _type=='train_epoch':
                print(log_note)
                top5_err=np.float(log_note.split('"top5_err": ')[1].split('}')[0]);#end with "}"
                loss=np.float(log_note.split('loss": ')[1].split(',')[0]);
                lr=np.float(log_note.split('lr": ')[1].split(',')[0]);
                train_top1_err.append(top1_err);
                train_top5_err.append(top5_err);
                train_lr.append(lr);
                train_epoch.append(epoch);
                train_loss.append(loss)
            elif _type=='val_epoch':
                top5_err=np.float(log_note.split('"top5_err": ')[1].split('}')[0]);#end with ","
                val_top1_err.append(top1_err);
                val_top5_err.append(top5_err);
                val_epoch.append(epoch);
    elif 'checkpoint.py: 214' in log_note:
            loaded_model.append(log_note);
            
# output_file.close()
    
plt.plot(train_epoch,train_loss,'b')
plt.axis([0,max(train_epoch), 0, max(train_loss)]) #xmin, xmax max(train_epoch), ymin, ymax
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

plt.figure()
plt.plot(val_epoch,val_top1_err,'g')
plt.xlabel('Epoch')
plt.ylabel('val_top1_err')
plt.axis([0, max(val_epoch), 0, max(val_top1_err)+1])
plt.show()

#plt.plot(train_epoch,train_top1_err,'r')
#plt.axis([0,  max(train_epoch), 0, max(train_top1_err)+5])
#plt.xlabel('Epoch')
#plt.ylabel('Training Error')
#plt.show()

    
