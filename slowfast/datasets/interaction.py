#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
import glob
import slowfast.utils.logging as logging

from . import decoder as decoder
from . import transform as transform
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from . import sampling
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Interaction(torch.utils.data.Dataset):
    """
    MECCANO video loader. Construct the MECCANO video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the MECCANO images loader with a given csv file. The format of 
        the csv file is:
        '''
        subj_1, video_id_1, task name1, R/L, label_name_1, label_id_1, frame_start_1, frame_end_1, bounding box 1
        subj_2, video_id_2, task name1, R/L, label_name_2, label_id_2, frame_start_2, frame_end_2,  bounding box 2
        ...
        subj_N, video_id_N, task nameN, R/L, label_name_N, label_id_N, frame_start_N, frame_end_N, bounding box N
        '''
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "test",
        ], "Split '{}' not supported for Interaction".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Constructing Interaction {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the data loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        #print("path_to_file:", path_to_file)
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        self._frame_start = []
        self._frame_end = []
        self._subj=[]
        self._vid=[]
        self._task_name=[]
        self._hand=[]
        self._label_name=[]
        self._bbx=[]
        self._frame_folder=[]
        self._head=[];
        self._flip=[];
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                # logger.info("clip_idx: {}".format(clip_idx));
                # logger.info("label seperate num: {}".format(len(path_label.split(','))));
                assert len(path_label.split(',')) == 11###
                subj, video_id, task_name, hand, label_name, label_id, GT, frame_start, frame_end, dset, flip= path_label.split(',')
               
                for idx in range(self._num_clips):
                    ##############
                    if dset=='HomeLab':                               
                        self._frame_folder.append('frames_HomeLab');
                        self._head.append('Dd');
                    elif dset=='Home':
                        self._frame_folder.append('frames_Home');
                        self._head.append('Ed');
                    
                    if len(subj)<2:
                        subj="0"+subj;                    
                    self._subj.append(subj);
                    
                    if len(video_id)<2:
                        video_id="0"+video_id;                                      
                    self._vid.append(video_id);
                    ##############
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_TO_DATA_DIR,self.cfg.DATA.PATH_PREFIX,self._frame_folder[clip_idx], subj, video_id))
                    
                    self._task_name.append(task_name)
                    self._hand.append(hand)
                    self._label_name.append(label_name)
                    # self._bbx.append([bounding_box_xmin,bounding_box_ymin,bounding_box_xmax,bounding_box_ymax])
                    self._frame_start.append(frame_start)
                    self._frame_end.append(frame_end)
                    self._labels.append(int(label_id))                    
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
                    self._flip.append(flip);
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Interaction split {} from {}"+path_to_file
        logger.info(
            "Constructing Interaction dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler.
        """
        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            # temporal_sample_index = 3#-1 ######## no need to ramdonly cropped images
            # spatial_sample_index = 3#-1 ######## no need to ramdonly cropped images
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            # temporal_sample_index = ( ######## no need to ramdonly cropped images
            #     self._spatial_temporal_idx[index]
            #     // self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            # # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # # center, or right if width is larger than height, and top, middle,
            # # or bottom if height is larger than width.
            # spatial_sample_index = ( ######## no need to ramdonly cropped images
            #     self._spatial_temporal_idx[index]
            #     % self.cfg.TEST.NUM_SPATIAL_CROPS
            # )
            min_scale, max_scale, crop_size = [self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )

        # Recover frames
        frames = [];
#        frames_test=[];
        
        hand=self._hand[index];
        For_missing_bbx=[];
        
        global bbx_R,bbx_L
        i=0;
        frame_count = int(self._frame_start[index]); 
        if int(self._frame_start[index])>int(self._frame_end[index]):#reverse instance
            
            while(frame_count >= int(self._frame_end[index])):
                i+=1;
            #rebuild original filename
                name_frame = str(frame_count)
                if(len(name_frame) == 5): #add a prefix 0
                    name_frame = "0"+name_frame
                elif(len(name_frame) == 4): #add a prefix 0
                    name_frame = "00"+name_frame
                elif(len(name_frame) == 3): #add two prefix 0
                    name_frame = "000"+name_frame
                elif(len(name_frame) == 2): #add three prefix 0
                    name_frame = "0000"+name_frame
                elif(len(name_frame) == 1): #add four prefix 0
                    name_frame = "00000"+name_frame
                
                framename=self._head[index]+self._subj[index]+'Cm'+self._vid[index]+'Hc'+name_frame;
                # logger.info("frame_folder: {}".format(self._frame_folder[index]))
                # logger.info("framename: {}".format(framename))
                for file in glob.glob(self.cfg.DATA.PATH_TO_DATA_DIR+'/'+self.cfg.DATA.PATH_PREFIX+'/'+self._frame_folder[index]+'/'+self._subj[index]+'/'+self._vid[index]+'/'+framename+'*.jpg'):#self.cfg.DATA.PATH_PREFIX
    #                logger.info("filename: {}".format(file))
                    image_name=file.split('/')[-1];
                    frame_count-=1 
                    image = Image.open(file)
                    #### get bbx file (under bbx folder)
                    bbx_file_path=glob.glob(self.cfg.DATA.PATH_TO_DATA_DIR+'/'+self.cfg.DATA.PATH_PREFIX+'/'+self._frame_folder[index]+'/'+self._subj[index]+'/'+self._vid[index]+'/Shan_bbx/'+framename+'*.txt')
                    # logger.info("bbx_file_path: {}".format(bbx_file_path))
                    if len(bbx_file_path)>0:# file found
                        with open(bbx_file_path[0],'r') as b:
                            listBox=[line.strip() for line in b]
                            bbx_R=[];bbx_L=[];
                        if len(listBox)>2:# has at least a hand
                            for j in range(2,len(listBox)+1):
                                if listBox[j-1][0]=="R":
                                    bbx_R=listBox[j-1].split('[')[1].strip(']').split(', ')
                                    for l in range(0,4):
                                        if int(bbx_R[l])<0:
                                            bbx_R[l]='0';
                                elif listBox[j-1][0]=="L":
                                    bbx_L=listBox[j-1].split('[')[1].strip(']').split(', ')
                                    for l in range(0,4):
                                        if int(bbx_L[l])<0:
                                            bbx_L[l]='0';
                            
                    else:#no txt
                        bbx_R=[];bbx_L=[];
                    
                    if hand=='R': 
                        bbx=bbx_R;
                    else:
                        bbx=bbx_L;
                    
                    if len(bbx)==0:# the hand had no bbx predicted
                        # logger.info("For_missing_bbx: {}".format(For_missing_bbx))
                        if len(For_missing_bbx)==0:#missing the 1st frame bbx
                            bbx=[0,0,0,0];
                        else:
                            bbx=For_missing_bbx;# use last available frame bbx
                    ### make bbx a square size
                    # logger.info("framename: {}".format(image_name))
                    # logger.info("bbx = {}".format(bbx))
                    bbx_size=[int(bbx[2])-int(bbx[0]),int(bbx[3])-int(bbx[1])];
                    if (int(bbx_size[0])<256) and (int(bbx[1])<256):
                        cropped_bbx=[int(bbx[0])-(256-bbx_size[1])/2,int(bbx[1])-(256-bbx_size[0])/2,int(bbx[2])+(256-bbx_size[0])/2,int(bbx[3])+(256-bbx_size[1])/2]
                    else:# resize and re-center
                        if int(bbx_size[0])>int(bbx_size[1]):
                            temp_size=int(bbx_size[1]);
                            cropped_bbx=[int(bbx[0])-(temp_size-bbx_size[0])/2,int(bbx[1]),int(bbx[2])+(temp_size-bbx_size[0])/2,int(bbx[3])];
                        else: # bbx_size[0]<=bbx_size[1]:
                            temp_size=int(bbx_size[0]);
                            cropped_bbx=[int(bbx[0]),int(bbx[1])-(temp_size-bbx_size[1])/2,int(bbx[2]),int(bbx[3])+(temp_size-bbx_size[1])/2];
                    # update latest bbx for missign frame
                    # if len(For_missing_bbx)==0:
                    #     For_missing_bbx.append(cropped_bbx)
                    # else: For_missing_bbx=cropped_bbx;
                    For_missing_bbx=cropped_bbx;
                    
                    image = image.crop([cropped_bbx[0],cropped_bbx[1],cropped_bbx[2],cropped_bbx[3]]) #### (left, top, right, bottom) #works
#                    logger.info('Flip: {}'.format(self._flip))
                    if self._flip[index]=='Flip':
                        image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
                    image = image.resize((256,256))
                    # image.save('/home/tsaim/anaconda3/envs/pyslowfast/slowfast/slowfast/Interaction_test/output/test_input/'+str(image_name)+'_reverse.jpg')####
                    image = np.asarray(image)
                    frames.append(torch.from_numpy(image))   
        else:
            while(frame_count <= int(self._frame_end[index])):
                i+=1;
            #rebuild original filename
                name_frame = str(frame_count)
                if(len(name_frame) == 5): #add a prefix 0
                    name_frame = "0"+name_frame
                elif(len(name_frame) == 4): #add a prefix 0
                    name_frame = "00"+name_frame
                elif(len(name_frame) == 3): #add two prefix 0
                    name_frame = "000"+name_frame
                elif(len(name_frame) == 2): #add three prefix 0
                    name_frame = "0000"+name_frame
                elif(len(name_frame) == 1): #add four prefix 0
                    name_frame = "00000"+name_frame
                
                framename=self._head[index]+self._subj[index]+'Cm'+self._vid[index]+'Hc'+name_frame;
                # logger.info("frame_folder: {}".format(self._frame_folder[index]))
                # logger.info("framename: {}".format(framename))
                for file in glob.glob(self.cfg.DATA.PATH_TO_DATA_DIR+'/'+self.cfg.DATA.PATH_PREFIX+'/'+self._frame_folder[index]+'/'+self._subj[index]+'/'+self._vid[index]+'/'+framename+'*.jpg'):#self.cfg.DATA.PATH_PREFIX
    #                logger.info("filename: {}".format(file))
                    image_name=file.split('/')[-1];
                    frame_count+=1 
                    image = Image.open(file)
                    #### get bbx file (under bbx folder)
                    bbx_file_path=glob.glob(self.cfg.DATA.PATH_TO_DATA_DIR+'/'+self.cfg.DATA.PATH_PREFIX+'/'+self._frame_folder[index]+'/'+self._subj[index]+'/'+self._vid[index]+'/Shan_bbx/'+framename+'*.txt')
                    # logger.info("bbx_file_path: {}".format(bbx_file_path))
                    if len(bbx_file_path)>0:# file found
                        with open(bbx_file_path[0],'r') as b:
                            listBox=[line.strip() for line in b]
                            bbx_R=[];bbx_L=[];
                        if len(listBox)>2:# has at least a hand
                            for j in range(2,len(listBox)+1):
                                if listBox[j-1][0]=="R":
                                    bbx_R=listBox[j-1].split('[')[1].strip(']').split(', ')
                                    for l in range(0,4):
                                        if int(bbx_R[l])<0:
                                            bbx_R[l]='0';
                                elif listBox[j-1][0]=="L":
                                    bbx_L=listBox[j-1].split('[')[1].strip(']').split(', ')
                                    for l in range(0,4):
                                        if int(bbx_L[l])<0:
                                            bbx_L[l]='0';
                    else:#no txt
                        bbx_R=[];bbx_L=[];
                            
                    if hand=='R': 
                        bbx=bbx_R;
                    else:
                        bbx=bbx_L;
                    
                    if len(bbx)==0:# the hand had no bbx predicted
                        # logger.info("For_missing_bbx: {}".format(For_missing_bbx))
                        if len(For_missing_bbx)==0:#missing the 1st frame bbx
                            bbx=[0,0,0,0];
                        else:
                            bbx=For_missing_bbx;# use last available frame bbx
                    ### make bbx a square size
                    # logger.info("framename: {}".format(image_name))
                    # logger.info("bbx = {}".format(bbx))
                    bbx_size=[int(bbx[2])-int(bbx[0]),int(bbx[3])-int(bbx[1])];
                    if (int(bbx_size[0])<256) and (int(bbx[1])<256):
                        cropped_bbx=[int(bbx[0])-(256-bbx_size[1])/2,int(bbx[1])-(256-bbx_size[0])/2,int(bbx[2])+(256-bbx_size[0])/2,int(bbx[3])+(256-bbx_size[1])/2]
                    else:# resize and re-center
                        if int(bbx_size[0])>int(bbx_size[1]):
                            temp_size=int(bbx_size[1]);
                            cropped_bbx=[int(bbx[0])-(temp_size-bbx_size[0])/2,int(bbx[1]),int(bbx[2])+(temp_size-bbx_size[0])/2,int(bbx[3])];
                        else: # bbx_size[0]<=bbx_size[1]:
                            temp_size=int(bbx_size[0]);
                            cropped_bbx=[int(bbx[0]),int(bbx[1])-(temp_size-bbx_size[1])/2,int(bbx[2]),int(bbx[3])+(temp_size-bbx_size[1])/2];
                    # update latest bbx for missign frame
                    # if len(For_missing_bbx)==0:
                    #     For_missing_bbx.append(cropped_bbx)
                    # else: For_missing_bbx=cropped_bbx;
                    For_missing_bbx=cropped_bbx;
                    
                    image = image.crop([cropped_bbx[0],cropped_bbx[1],cropped_bbx[2],cropped_bbx[3]]) #### (left, top, right, bottom) #works
#                    logger.info('Flip: {}'.format(self._flip))
                    if self._flip[index]=='Flip':
                        image = image.transpose(method=Image.FLIP_LEFT_RIGHT)
                    image = image.resize((256,256))
                    # image.save('/home/tsaim/anaconda3/envs/pyslowfast/slowfast/slowfast/Interaction_test/output/test_input/'+str(image_name)+'.jpg')####
                    image = np.asarray(image)
                    frames.append(torch.from_numpy(image))   
                    
        frames= torch.stack(frames)
        #sampling frames
        if  int(self._frame_start[index])>int(self._frame_end[index]):#reverse instance
            frames = sampling.temporal_sampling(frames, int(self._frame_end[index]), int(self._frame_start[index]), self.cfg.DATA.NUM_FRAMES)
        else:
            frames = sampling.temporal_sampling(frames, int(self._frame_start[index]), int(self._frame_end[index]), self.cfg.DATA.NUM_FRAMES)
        
        # Perform color normalization.
        frames = frames / 255.0
        frames = frames - torch.tensor(self.cfg.DATA.MEAN)
        frames = frames / torch.tensor(self.cfg.DATA.STD) 
#        logger.info("Number of frames in the instance: {}".format(frames.size(0)))       
        
        # T H W C -> C T H W.
        frames = frames.permute(3, 0, 1, 2)  

        #Perform data augmentation ### no need to randomly cropped images
        # frames = self.spatial_sampling(
        #         frames,
        #         spatial_idx=spatial_sample_index,
        #         min_scale=min_scale,
        #         max_scale=max_scale,
        #         crop_size=crop_size,
        #         random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
        #     )
        
        label = self._labels[index]
        # pack_pathway_output: return a list of tensors with the dimension of `channel` x `num frames` x `height` x `width`.
        frames = utils.pack_pathway_output(self.cfg, frames) 
        # logger.info("Number of Chennel: {}".format(frames[0].size(0))) 
        
        return image_name, hand, frames, label, index, {}
        ###########################################

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)

    def spatial_sampling(
        self,
        frames,
        spatial_idx=-1,
        min_scale=256,
        max_scale=320,
        crop_size=224,
        random_horizontal_flip=True,
    ):
        """
        Perform spatial sampling on the given video frames. If spatial_idx is
        -1, perform random scale, random crop, and random flip on the given
        frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
        with the given spatial_idx.
        Args:
            frames (tensor): frames of images sampled from the video. The
                dimension is `num frames` x `height` x `width` x `channel`.
            spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
                or 2, perform left, center, right crop if width is larger than
                height, and perform top, center, buttom crop if height is larger
                than width.
            min_scale (int): the minimal size of scaling.
            max_scale (int): the maximal size of scaling.
            crop_size (int): the size of height and width used to crop the
                frames.
        Returns:
            frames (tensor): spatially sampled frames.
        """
        assert spatial_idx in [-1, 0, 1, 2]
        if spatial_idx == -1:
            frames, _ = transform.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
            )
            frames, _ = transform.random_crop(frames, crop_size)
            if random_horizontal_flip:
                frames, _ = transform.horizontal_flip(0.5, frames)
        else:
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale, crop_size}) == 1
            frames, _ = transform.random_short_side_scale_jitter(
                frames, min_scale, max_scale
            )
            frames, _ = transform.uniform_crop(frames, crop_size, spatial_idx)
        return frames
    
#################### added 
    @property
    def num_videos(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)