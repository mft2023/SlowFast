# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 15:15:41 2021

@author: tsaiM
"""
import pickle as pk
import numpy as np
import scipy.io as sio
import re
import glob
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
import copy

def atoi(text):
    return int(text) if text.isdigit() else text
    

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def find_subj_in_list(A,B):#A:subj,frame; B:list
    num=[]
    for i in range(0,len(B)):
        if B[i]==A:
          num.append(i)
    return num

def find_in_list(A,B):#A:subj,frame; B:list
    num=[]
    for i in range(0,len(B)):
        if B[i][0]==A:
          num.append(i)
    return num

#get target list and participant info
target_tasks = pd.read_excel('{path to the excel file with all analyzed tasks/frames}/TargetLabeling_Home.xlsx',header=None);
target_info=target_tasks.iloc[3:75,0:6] #each raw includes: task name, task type:bilateral(b) or one-handed(o) task, sub, start frame, end frame, video set 
target_list=target_info.values.tolist()

# For interaction:
#num_frames_per_instance=32;
#label_col=[0,1];dataset='Interaction';
#w_interaction_frame='';#get every frame

# For hand roles:
num_frames_per_instance=16;
label_col=[4,5];dataset='Manipulation';
w_interaction_frame='on';# get results from frames with interaction only

file_folder='{path to results folder}/'+dataset+'/Home/';
file_path=glob.glob(file_folder+'/Results/*.pickle');

saveRoot=file_folder;
Participant_info=pd.read_excel('{path to the file that includes participant id and their affected side}/participants_info.xlsx',sheet_name='demographic');
GT_labeling_Folder_HomeLab='{path to HomeLab dataset annotations}/Documents for Home Lab/Labeled xlsx/';
GT_labeling_Folder_Home='{path to Home dataset annotations}/Documents for Home/Labeled xlsx/';

# read participants information to find Aff/Uaff hand
Aff_Hand_info=Participant_info.iloc[:,3] #sub,affted hand (R=1;L=0)
Aff_Hand_list=Aff_Hand_info.values.tolist()

##################################### SlowFast Results (Instance-wise) ################################
Tested_filename_hand_pred_slowfastGT=[];
for m in range(0,len(file_path)):
    results_file_path=file_path[m];     
    results_data=[];
    with open(results_file_path, 'rb') as f:
        results_data.append(pk.load(f))
    
    # Organize the results (0:[filename, hand], 1:[non-manipulation score, manipulation score], 2:[labels])
    filename=results_data[0][0];
    prediction_confidence=results_data[0][1].numpy(); #[non-manipulation score, manipulation score]
    slowfast_labels=results_data[0][2].numpy();
    
    prediction=[];
    for i in range(0,len(prediction_confidence)):
        non_interaction=prediction_confidence[i][0];
        interaction=prediction_confidence[i][1];
        if interaction>non_interaction:
            prediction.append(1);
        else:
            prediction.append(0);
    
    for j in range(0,len(filename)):
        Tested_filename_hand_pred_slowfastGT.append([filename[j][0], filename[j][1], prediction[j], slowfast_labels[j]]);
        

subj_list=[];
subj_aff_slowfast_pred=[];subj_unaff_slowfast_pred=[];subj_overall_slowfast_pred=[];
subj_aff_slowfast_GT=[];subj_unaff_slowfast_GT=[];subj_overall_slowfast_GT=[];
Subj_aff_frame_pred_GT=[];Subj_unaff_frame_pred_GT=[];Subj_overall_frame_pred_GT=[];
#match subj, video set, frame#, and hand side
for i in range(0,len(target_list)):
    task_subj=target_list[i][2];
    print('Task Subj: '+str(task_subj))
    if [str(task_subj)] not in subj_list:
        subj_list.append([str(task_subj)]);
        subj_aff_slowfast_pred.extend([[str(task_subj)]]);
        subj_aff_slowfast_GT.extend([[str(task_subj)]]);
        subj_unaff_slowfast_pred.extend([[str(task_subj)]]);
        subj_unaff_slowfast_GT.extend([[str(task_subj)]]);
        subj_overall_slowfast_pred.extend([[str(task_subj)]]);
        subj_overall_slowfast_GT.extend([[str(task_subj)]]);
    
    subj_loc_labeling=find_subj_in_list(task_subj,list(Participant_info.iloc[:,0]));
    aff_hand_num=Aff_Hand_list[subj_loc_labeling[0]];#R:1, L:0
    if aff_hand_num==1:
        aff_hand='R';
    else:
        aff_hand='L';

    task_vset=target_list[i][5];
    task_start_frame=target_list[i][3];
    task_end_frame=target_list[i][4];
    GT_label=[];
    for j in range(0,len(Tested_filename_hand_pred_slowfastGT)):
        tested_subj=Tested_filename_hand_pred_slowfastGT[j][0][2:4];
        tested_vset=Tested_filename_hand_pred_slowfastGT[j][0][6:8];
        tested_frame=int(Tested_filename_hand_pred_slowfastGT[j][0][10:16]);#last frame in the instance
        tested_hand=Tested_filename_hand_pred_slowfastGT[j][1];        
        pred=int(Tested_filename_hand_pred_slowfastGT[j][2]);
        slowfast_GT=int(Tested_filename_hand_pred_slowfastGT[j][3]);#label
        
    ######################################### for frame-wise results ###################################################            
        if (task_subj==int(tested_subj)) and (task_vset==int(tested_vset)) and (tested_frame>=task_start_frame) and (tested_frame<=task_end_frame):#same task
            subj_loc=find_subj_in_list(str(task_subj),np.array(subj_list)); 
            if Tested_filename_hand_pred_slowfastGT[j][0][0:2]=='Dd':
                dset='HomeLab';
            else:
                dset='Home';
            #read GT from labelig file
            if len(GT_label)==0:#not read the task labeling yet
                GT_filename ='labeled_stroke_'+dset+'_sub' + tested_subj+ '_' + tested_vset + '.xlsx';
                if dset=='Home':
                    GT_label = pd.read_excel(GT_labeling_Folder_Home+GT_filename,header=None);
                elif dset=='HomeLab':                        
                    GT_label = pd.read_excel(GT_labeling_Folder_HomeLab+GT_filename,header=None);
            
            frames=range(tested_frame-num_frames_per_instance,tested_frame);# frames in an instance
            
            if aff_hand==tested_hand:#affected hand
                subj_aff_slowfast_pred[subj_loc[0]].extend([pred]);
                subj_aff_slowfast_GT[subj_loc[0]].extend([slowfast_GT]);
                for k in range(num_frames_per_instance):
                    frame=frames[k];
                    if (frame>=task_start_frame) and (frame<=task_end_frame):
                        current_pred=copy.deepcopy(pred);
                        index=find_in_list([tested_subj, tested_vset,frame,tested_hand],Subj_aff_frame_pred_GT);
                        if len(index)>0: #tested task in the target list
                            Subj_aff_frame_pred_GT[index[0]][1].append(current_pred);
                        else:
                            final_pred=copy.deepcopy(current_pred);
                            if tested_hand=='R':
                                if w_interaction_frame=='on':
                                    interaction_label=int(GT_label.iloc[frame-1 , 0]);
                                    if interaction_label==1:#ongoing
                                        GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                        if GT==2:# no hand in the frame: label as non-manipulation
                                            GT=0;
                                        Subj_aff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                else:#doesn't matter wheteher it's an interaction frame                                            
                                    GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                    if GT==2:
                                        GT=0;
                                    Subj_aff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                            elif tested_hand=='L':# L hand
                                if w_interaction_frame=='on':
                                    interaction_label=int(GT_label.iloc[frame-1 , 1]);
                                    if interaction_label==1:
                                        GT=int(GT_label.iloc[frame-1 , label_col[1]]);
                                        if GT==2:
                                            GT=0;
                                        Subj_aff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                else:#doesn't matter wheteher it's an interaction frame                                            
                                    GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                    if GT==2:
                                        GT=0;
                                    Subj_aff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
            else:# non-affected hand
                subj_unaff_slowfast_pred[subj_loc[0]].extend([pred]);
                subj_unaff_slowfast_GT[subj_loc[0]].extend([slowfast_GT]);
                for k in range(num_frames_per_instance):
                    frame=frames[k];
                    if (frame>=task_start_frame) and (frame<=task_end_frame):
                        current_pred=copy.deepcopy(pred);                    
                        index=find_in_list([tested_subj,tested_vset,frame,tested_hand],Subj_unaff_frame_pred_GT);
                        if len(index)>0: # in the list
                           Subj_unaff_frame_pred_GT[index[0]][1].append(current_pred);
                        else:
                            final_pred=copy.deepcopy(current_pred);
                            if tested_hand=='R':       
                                if w_interaction_frame=='on':
                                    interaction_label=int(GT_label.iloc[frame-1 , 0]);
                                    if interaction_label==1:
                                        GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                        if GT==2:
                                            GT=0;
                                        Subj_unaff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                else:#doesn't matter wheteher it's an interaction frame                                            
                                    GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                    if GT==2:
                                        GT=0;
                                    Subj_unaff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                    
                            elif tested_hand=='L':# L hand
                                if w_interaction_frame=='on':
                                    interaction_label=int(GT_label.iloc[frame-1 , 1]);
                                    if interaction_label==1:
                                        GT=int(GT_label.iloc[frame-1 , label_col[1]]);
                                        if GT==2:
                                            GT=0;
                                        Subj_unaff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                else:#doesn't matter wheteher it's an interaction frame                                            
                                    GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                    if GT==2:
                                        GT=0;
                                    Subj_unaff_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                    
            subj_overall_slowfast_pred[subj_loc[0]].extend([pred]);
            subj_overall_slowfast_GT[subj_loc[0]].extend([slowfast_GT]);
            
            for k in range(num_frames_per_instance):
                frame=frames[k];
                if (frame>=task_start_frame) and (frame<=task_end_frame):
                    current_pred=copy.deepcopy(pred);                    
                    index=find_in_list([tested_subj,tested_vset,frame,tested_hand],Subj_overall_frame_pred_GT);
                    if len(index)>0: 
                        Subj_overall_frame_pred_GT[index[0]][1].append(current_pred);
                    else:
                        final_pred=copy.deepcopy(current_pred);
                        if tested_hand=='R':
                            if w_interaction_frame=='on':
                                interaction_label=int(GT_label.iloc[frame-1 , 0]);
                                if interaction_label==1:
                                    GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                    if GT==2:
                                        GT=0;
                                    Subj_overall_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                            else:#doesn't matter wheteher it's an interaction frame                                            
                                GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                if GT==2:
                                    GT=0;
                                Subj_overall_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                
                        elif tested_hand=='L':# L hand
                            if w_interaction_frame=='on':
                                interaction_label=int(GT_label.iloc[frame-1 , 1]);
                                if interaction_label==1:
                                    GT=int(GT_label.iloc[frame-1 , label_col[1]]);
                                    if GT==2:
                                        GT=0;
                                    Subj_overall_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                            else:#doesn't matter wheteher it's an interaction frame                                            
                                GT=int(GT_label.iloc[frame-1 , label_col[0]]);
                                if GT==2:
                                    GT=0;
                                Subj_overall_frame_pred_GT.append([[tested_subj,tested_vset,frame,tested_hand],[final_pred], GT]);
                                
                           
 
#Calculate F1-score, precision, recal, and accuracy
F1_prec_recall_ACC_aff=[];matthews_aff=[];
F1_prec_recall_ACC_unaff=[];matthews_unaff=[];
F1_prec_recall_ACC_overall=[];matthews_overall=[];
overall_ACC=[];overall_F1=[];overall_prec=[];overall_recall=[];overall_matthews=[];
aff_ACC=[];aff_F1=[];aff_prec=[];aff_recall=[];aff_matthews=[];
unaff_ACC=[];unaff_F1=[];unaff_prec=[];unaff_recall=[];unaff_matthews=[];
num_int=0;num_nonint=0;All_aff_GT=[];All_aff_pred=[];All_unaff_GT=[];All_unaff_pred=[];All_overall_GT=[];All_overall_pred=[];
for k in range(len(subj_list)):
    subj=subj_aff_slowfast_GT[k][0];
    print('subj: '+subj)
    aff_pred=subj_aff_slowfast_pred[k][1:];
    aff_GT=subj_aff_slowfast_GT[k][1:];
    unaff_pred=subj_unaff_slowfast_pred[k][1:];
    unaff_GT=subj_unaff_slowfast_GT[k][1:];
    overall_pred=subj_overall_slowfast_pred[k][1:];
    overall_GT=subj_overall_slowfast_GT[k][1:];
    num_int=num_int+len(np.where(np.array(overall_GT,dtype=np.int8)==1)[0]);
    num_nonint=num_nonint+len(np.where(np.array(overall_GT,dtype=np.int8)==0)[0]);
    
    aff_F1.append(f1_score(aff_GT,aff_pred));
    aff_ACC.append(accuracy_score(aff_GT,aff_pred))
    aff_prec.append(precision_score(aff_GT,aff_pred))
    aff_recall.append(recall_score(aff_GT,aff_pred))
    aff_matthews.append(matthews_corrcoef(aff_GT,aff_pred))
    
    unaff_F1.append(f1_score(unaff_GT,unaff_pred));
    unaff_ACC.append(accuracy_score(unaff_GT,unaff_pred))
    unaff_prec.append(precision_score(unaff_GT,unaff_pred))
    unaff_recall.append(recall_score(unaff_GT,unaff_pred))
    unaff_matthews.append(matthews_corrcoef(unaff_GT,unaff_pred))
    
    overall_F1.append(f1_score(overall_GT,overall_pred));
    overall_ACC.append(accuracy_score(overall_GT,overall_pred));
    overall_prec.append(precision_score(overall_GT,overall_pred))
    overall_recall.append(recall_score(overall_GT,overall_pred))
    overall_matthews.append(matthews_corrcoef(overall_GT,overall_pred))
    
    matthews_aff.append([subj,matthews_corrcoef(aff_GT,aff_pred)]);
    matthews_unaff.append([subj,matthews_corrcoef(unaff_GT,unaff_pred)]);
    matthews_overall.append([subj,matthews_corrcoef(overall_GT,overall_pred)]);
    F1_prec_recall_ACC_aff.append([subj,f1_score(aff_GT,aff_pred), precision_score(aff_GT,aff_pred), recall_score(aff_GT,aff_pred), accuracy_score(aff_GT,aff_pred)]);
    F1_prec_recall_ACC_unaff.append([subj,f1_score(unaff_GT,unaff_pred), precision_score(unaff_GT,unaff_pred), recall_score(unaff_GT,unaff_pred), accuracy_score(unaff_GT,unaff_pred)]);
    F1_prec_recall_ACC_overall.append([subj,f1_score(overall_GT,overall_pred), precision_score(overall_GT,overall_pred), recall_score(overall_GT,overall_pred), accuracy_score(overall_GT,overall_pred)]);
    
    ###### micro #########
    All_aff_GT=All_aff_GT+aff_GT;
    All_aff_pred=All_aff_pred+aff_pred;
    All_unaff_GT=All_unaff_GT+unaff_GT;
    All_unaff_pred=All_unaff_pred+unaff_pred;
    All_overall_GT=All_overall_GT+overall_GT;
    All_overall_pred=All_overall_pred+overall_pred;
    

print('\n======= Unfiltered and Flipped Instance-Wise Results =======')
print('Total number of /instance/: ',str((num_int+num_nonint)))
print(dataset,' /instance/: ',str(round(num_int/(num_int+num_nonint)*100,2)),' %')
print('\n======== Affected Hand ========')
print('Average MCC (By Subj): ', round(np.mean(aff_matthews),2), ' +- ', round(np.std(aff_matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(aff_F1),2), ' +- ', round(np.std(aff_F1),2))
print('Average Precision (By Subj): ', round(np.mean(aff_prec),2), ' +- ', round(np.std(aff_prec),2))
print('Average Recall (By Subj): ', round(np.mean(aff_recall),2), ' +- ', round(np.std(aff_recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(aff_ACC),2), ' +- ', round(np.std(aff_ACC),2))

print('\n======= Unaffected Hand =======')
print('Average MCC (By Subj): ', round(np.mean(unaff_matthews),2), ' +- ', round(np.std(unaff_matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(unaff_F1),2), ' +- ', round(np.std(unaff_F1),2))   
print('Average Precision (By Subj): ', round(np.mean(unaff_prec),2), ' +- ', round(np.std(unaff_prec),2))
print('Average Recall (By Subj): ', round(np.mean(unaff_recall),2), ' +- ', round(np.std(unaff_recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(unaff_ACC),2), ' +- ', round(np.std(unaff_ACC),2)) 

print('\n=========== Overall ===========')
print('Average MCC (By Subj): ', round(np.mean(overall_matthews),2), ' +- ', round(np.std(overall_matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(overall_F1),2), ' +- ', round(np.std(overall_F1),2))
print('Average Precision (By Subj): ', round(np.mean(overall_prec),2), ' +- ', round(np.std(overall_prec),2))
print('Average Recall (By Subj): ', round(np.mean(overall_recall),2), ' +- ', round(np.std(overall_recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(overall_ACC),2), ' +- ', round(np.std(overall_ACC),2))


########### Micro Average ###########
print('\n======== Micro Average ========')
print('======== Affected Hand ========')
print('Average MCC: ', matthews_corrcoef(All_aff_GT,All_aff_pred))
print('Average F1 Score: ', f1_score(All_aff_GT,All_aff_pred))
print('Average Precision: ', precision_score(All_aff_GT,All_aff_pred))
print('Average Recall: ', recall_score(All_aff_GT,All_aff_pred))
print('Average Accuracy Score: ' , accuracy_score(All_aff_GT,All_aff_pred),'\n')

print('\n======= Unaffected Hand =======')
print('Average MCC: ', matthews_corrcoef(All_unaff_GT,All_unaff_pred))
print('Average F1 Score: ', f1_score(All_unaff_GT,All_unaff_pred))
print('Average Precision: ', precision_score(All_unaff_GT,All_unaff_pred))
print('Average Recall: ', recall_score(All_unaff_GT,All_unaff_pred))
print('Average Accuracy Score: ' , accuracy_score(All_unaff_GT,All_unaff_pred),'\n')

print('\n=========== Overall ===========')
print('Average MCC: ', matthews_corrcoef(All_overall_GT,All_overall_pred))
print('Average F1 Score: ', f1_score(All_overall_GT,All_overall_pred))
print('Average Precision: ', precision_score(All_overall_GT,All_overall_pred))
print('Average Recall: ', recall_score(All_overall_GT,All_overall_pred))
print('Average Accuracy Score: ' , accuracy_score(All_overall_GT,All_overall_pred),'\n')

##################################### SlowFast Results (frame-wise) ################################   
#decide final prediction using mode
for i in range(0,len(Subj_aff_frame_pred_GT)):
    final_pred=np.mean(Subj_aff_frame_pred_GT[i][1]);
    if final_pred<0.5:
        final_pred=0;
    else:
        final_pred=1;
    Subj_aff_frame_pred_GT[i][1]=final_pred;

for i in range(0,len(Subj_unaff_frame_pred_GT)):
    final_pred=np.mean(Subj_unaff_frame_pred_GT[i][1]);
    if final_pred<0.5:
        final_pred=0;
    else:
        final_pred=1;
    Subj_unaff_frame_pred_GT[i][1]=final_pred;
    
for i in range(0,len(Subj_overall_frame_pred_GT)):
    final_pred=np.mean(Subj_overall_frame_pred_GT[i][1]);
    if final_pred<0.5:
        final_pred=0;
    else:
        final_pred=1;
    Subj_overall_frame_pred_GT[i][1]=final_pred;
    
# Organize data by subj
subj_aff_pred=copy.deepcopy(subj_list);subj_aff_GT=copy.deepcopy(subj_list);
subj_unaff_pred=copy.deepcopy(subj_list);subj_unaff_GT=copy.deepcopy(subj_list);
subj_overall_pred=copy.deepcopy(subj_list);subj_overall_GT=copy.deepcopy(subj_list);
for i in range(0,len(Subj_aff_frame_pred_GT)):
    subj=find_subj_in_list(str(int(Subj_aff_frame_pred_GT[i][0][0])),np.array(subj_list));
    ind=int(subj[0]);
    subj_aff_pred[ind].extend([Subj_aff_frame_pred_GT[i][1]]);
    subj_aff_GT[ind].extend([Subj_aff_frame_pred_GT[i][2]]);
    
for i in range(0,len(Subj_unaff_frame_pred_GT)):
    subj=find_subj_in_list(str(int(Subj_unaff_frame_pred_GT[i][0][0])),np.array(subj_list));
    ind=int(subj[0]);
    subj_unaff_pred[ind].extend([Subj_unaff_frame_pred_GT[i][1]]);
    subj_unaff_GT[ind].extend([Subj_unaff_frame_pred_GT[i][2]]);

for i in range(0,len(Subj_overall_frame_pred_GT)):
    subj=find_subj_in_list(str(int(Subj_overall_frame_pred_GT[i][0][0])),np.array(subj_list));
    ind=int(subj[0]);
    subj_overall_pred[ind].extend([Subj_overall_frame_pred_GT[i][1]]);
    subj_overall_GT[ind].extend([Subj_overall_frame_pred_GT[i][2]]);
             

F1_prec_recall_ACC_aff=[];matthews_aff=[];
F1_prec_recall_ACC_unaff=[];matthews_unaff=[];
F1_prec_recall_ACC_overall=[];matthews_overall=[];
overall_ACC=[];overall_F1=[];overall_prec=[];overall_recall=[];overall_matthews=[]
aff_ACC=[];aff_F1=[];aff_prec=[];aff_recall=[];aff_matthews=[];
unaff_ACC=[];unaff_F1=[];unaff_prec=[];unaff_recall=[];unaff_matthews=[];
micro_matthews_overall=[];micro_F1_overall=[];micro_prec_overall=[];micro_recall_overall=[];micro_ACC_overall=[];
micro_matthews_aff=[];micro_F1_aff=[];micro_prec_aff=[];micro_recall_aff=[];micro_ACC_aff=[];
micro_matthews_unaff=[];micro_F1_unaff=[];micro_prec_unaff=[];micro_recall_unaff=[];micro_ACC_unaff=[];
num_int=0;num_nonint=0;All_aff_GT=[];All_aff_pred=[];All_unaff_GT=[];All_unaff_pred=[];All_overall_GT=[];All_overall_pred=[];
for k in range(len(subj_list)):
    subj=subj_aff_pred[k][0];
    aff_pred=subj_aff_pred[k][1:];
    aff_GT=subj_aff_GT[k][1:];
    unaff_pred=subj_unaff_pred[k][1:];
    unaff_GT=subj_unaff_GT[k][1:];
    overall_pred=subj_overall_pred[k][1:];
    overall_GT=subj_overall_GT[k][1:];
    
    num_int=num_int+len(np.where(np.array(overall_GT,dtype=np.int8)==1)[0]);
    num_nonint=num_nonint+len(np.where(np.array(overall_GT,dtype=np.int8)==0)[0]);
    
    aff_matthews.append(matthews_corrcoef(aff_GT,aff_pred));
    aff_F1.append(f1_score(aff_GT,aff_pred));
    aff_ACC.append(accuracy_score(aff_GT,aff_pred))
    aff_prec.append(precision_score(aff_GT,aff_pred))
    aff_recall.append(recall_score(aff_GT,aff_pred))
    
    unaff_matthews.append(matthews_corrcoef(unaff_GT,unaff_pred));
    unaff_F1.append(f1_score(unaff_GT,unaff_pred));
    unaff_ACC.append(accuracy_score(unaff_GT,unaff_pred))
    unaff_prec.append(precision_score(unaff_GT,unaff_pred))
    unaff_recall.append(recall_score(unaff_GT,unaff_pred))
    
    overall_matthews.append(matthews_corrcoef(overall_GT,overall_pred));
    overall_F1.append(f1_score(overall_GT,overall_pred));
    overall_ACC.append(accuracy_score(overall_GT,overall_pred));
    overall_prec.append(precision_score(overall_GT,overall_pred))
    overall_recall.append(recall_score(overall_GT,overall_pred))
    
    
    ###### micro #########
    All_aff_GT=All_aff_GT+aff_GT;
    All_aff_pred=All_aff_pred+aff_pred;
    All_unaff_GT=All_unaff_GT+unaff_GT;
    All_unaff_pred=All_unaff_pred+unaff_pred;
    All_overall_GT=All_overall_GT+overall_GT;
    All_overall_pred=All_overall_pred+overall_pred;
    
    matthews_aff.append([subj,matthews_corrcoef(aff_GT,aff_pred)]);
    matthews_unaff.append([subj,matthews_corrcoef(unaff_GT,unaff_pred)]);
    matthews_overall.append([subj,matthews_corrcoef(overall_GT,overall_pred)]);
    F1_prec_recall_ACC_aff.append([subj,f1_score(aff_GT,aff_pred), precision_score(aff_GT,aff_pred), recall_score(aff_GT,aff_pred), accuracy_score(aff_GT,aff_pred)]);
    F1_prec_recall_ACC_unaff.append([subj,f1_score(unaff_GT,unaff_pred), precision_score(unaff_GT,unaff_pred), recall_score(unaff_GT,unaff_pred), accuracy_score(unaff_GT,unaff_pred)]);
    F1_prec_recall_ACC_overall.append([subj,f1_score(overall_GT,overall_pred), precision_score(overall_GT,overall_pred), recall_score(overall_GT,overall_pred), accuracy_score(overall_GT,overall_pred)]);

print('\n======= Unfiltered and Flipped Frame-Wise Results =======')
print('Total number of frame instance: ',str((num_int+num_nonint)))#### two hands combined
print(dataset,' /frames/: ',str(round(num_int/(num_int+num_nonint)*100,2)),' %')
print('\n======== Affected Hand ========')
print('Average MCC (By Subj): ', round(np.mean(aff_matthews),2), ' +- ', round(np.std(aff_matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(aff_F1),2), ' +- ', round(np.std(aff_F1),2))
print('Average Precision (By Subj): ', round(np.mean(aff_prec),2), ' +- ', round(np.std(aff_prec),2))
print('Average Recall (By Subj): ', round(np.mean(aff_recall),2), ' +- ', round(np.std(aff_recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(aff_ACC),2), ' +- ', round(np.std(aff_ACC),2))

print('\n======= Unaffected Hand =======')
print('Average MCC (By Subj): ', round(np.mean(unaff_matthews),2), ' +- ', round(np.std(unaff_matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(unaff_F1),2), ' +- ', round(np.std(unaff_F1),2))   
print('Average Precision (By Subj): ', round(np.mean(unaff_prec),2), ' +- ', round(np.std(unaff_prec),2))
print('Average Recall (By Subj): ', round(np.mean(unaff_recall),2), ' +- ', round(np.std(unaff_recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(unaff_ACC),2), ' +- ', round(np.std(unaff_ACC),2)) 

print('\n=========== Overall ===========')
print('Average MCC (By Subj): ', round(np.mean(overall_matthews),2), ' +- ', round(np.std(overall_matthews),2))
print('Average F1 Score (By Subj): ', round(np.mean(overall_F1),2), ' +- ', round(np.std(overall_F1),2))
print('Average Precision (By Subj): ', round(np.mean(overall_prec),2), ' +- ', round(np.std(overall_prec),2))
print('Average Recall (By Subj): ', round(np.mean(overall_recall),2), ' +- ', round(np.std(overall_recall),2))
print('Average Accuracy Score (By Subj): ' , round(np.mean(overall_ACC),2), ' +- ', round(np.std(overall_ACC),2))

########### Micro Average ###########
print('\n======== Micro Average ========')
print('======== Affected Hand ========')
print('Average MCC: ', matthews_corrcoef(All_aff_GT,All_aff_pred))
print('Average F1 Score: ', f1_score(All_aff_GT,All_aff_pred))
print('Average Precision: ', precision_score(All_aff_GT,All_aff_pred))
print('Average Recall: ', recall_score(All_aff_GT,All_aff_pred))
print('Average Accuracy Score: ' , accuracy_score(All_aff_GT,All_aff_pred),'\n')

print('\n======= Unaffected Hand =======')
print('Average MCC: ', matthews_corrcoef(All_unaff_GT,All_unaff_pred))
print('Average F1 Score: ', f1_score(All_unaff_GT,All_unaff_pred))
print('Average Precision: ', precision_score(All_unaff_GT,All_unaff_pred))
print('Average Recall: ', recall_score(All_unaff_GT,All_unaff_pred))
print('Average Accuracy Score: ' , accuracy_score(All_unaff_GT,All_unaff_pred),'\n')

print('\n=========== Overall ===========')
print('Average MCC: ', matthews_corrcoef(All_overall_GT,All_overall_pred))
print('Average F1 Score: ', f1_score(All_overall_GT,All_overall_pred))
print('Average Precision: ', precision_score(All_overall_GT,All_overall_pred))
print('Average Recall: ', recall_score(All_overall_GT,All_overall_pred))
print('Average Accuracy Score: ' , accuracy_score(All_overall_GT,All_overall_pred),'\n')

s3 = saveRoot + '/LOSOCV_Home_SlowFast_'+dataset+'_overall_results.mat'
s4 = saveRoot + '/LOSOCV_Home_SlowFast_'+dataset+'_aff_results.mat' 
s5 = saveRoot + '/LOSOCV_Home_SlowFast_'+dataset+'_unaff_results.mat' 
                    
sio.savemat(s3, {'matthews_overall':matthews_overall,'SUBJ_overall_F1_prec_recall_ACC':F1_prec_recall_ACC_overall,'SUBJ_overall_GT':subj_overall_GT,'SUBJ_overall_pred':subj_overall_pred})
sio.savemat(s4, {'matthews_aff':matthews_aff,'SUBJ_aff_F1_prec_recall_ACC':F1_prec_recall_ACC_aff,'SUBJ_aff_GT':subj_aff_GT,'SUBJ_aff_pred':subj_aff_pred})
sio.savemat(s5, {'matthews_unaff':matthews_unaff,'SUBJ_unaff_F1_prec_recall_ACC':F1_prec_recall_ACC_unaff,'SUBJ_unaff_GT':subj_unaff_GT,'SUBJ_unaff_pred':subj_unaff_pred})