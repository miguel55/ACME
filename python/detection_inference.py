#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:41:34 2019

@author: mmolina
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import scipy.io as sio
import time
import torch.nn.functional as FT
import natsort
from config import cfg
import torchvision.transforms.functional as F
from Joint3DSegmentationModule import net as ACMENet
from sklearn.metrics import roc_auc_score
import abc
from scipy.ndimage.measurements import label as lb

# Segmentation to bounding box utils.
# Adapted from medicaldetectionToolkit
# https://github.com/MIC-DKFZ/medicaldetectiontoolkit
# [3] Jaeger, Paul et al. "Retina U-Net: Embarrassingly Simple Exploitation of 
# Segmentation Supervision for Medical Object Detection" , 2018 

# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

class AbstractTransform(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self, **data_dict):
        raise NotImplementedError("Abstract, so implement")

    def __repr__(self):
        ret_str = str(type(self).__name__) + "( " + ", ".join(
            [key + " = " + repr(val) for key, val in self.__dict__.items()]) + " )"
        return ret_str
    
class ConvertSegToBoundingBoxCoordinates(AbstractTransform):
    """ Converts segmentation masks into bounding box coordinates.
    """

    def __init__(self, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):
        self.dim = dim
        self.get_rois_from_seg_flag = get_rois_from_seg_flag
        self.class_specific_seg_flag = class_specific_seg_flag

    def __call__(self, **data_dict):
        data_dict = convert_seg_to_bounding_box_coordinates(data_dict, self.dim, self.get_rois_from_seg_flag, class_specific_seg_flag=self.class_specific_seg_flag)
        return data_dict
    
def convert_seg_to_bounding_box_coordinates(data_dict, dim, get_rois_from_seg_flag=False, class_specific_seg_flag=False):

    '''
    This function generates bounding box annotations from given pixel-wise annotations.
    :param data_dict: Input data dictionary as returned by the batch generator.
    :param dim: Dimension in which the model operates (2 or 3).
    :param get_rois_from_seg: Flag specifying one of the following scenarios:
    1. A label map with individual ROIs identified by increasing label values, accompanied by a vector containing
    in each position the class target for the lesion with the corresponding label (set flag to False)
    2. A binary label map. There is only one foreground class and single lesions are not identified.
    All lesions have the same class target (foreground). In this case the Dataloader runs a Connected Component
    Labelling algorithm to create processable lesion - class target pairs on the fly (set flag to True).
    :param class_specific_seg_flag: if True, returns the pixelwise-annotations in class specific manner,
    e.g. a multi-class label map. If False, returns a binary annotation map (only foreground vs. background).
    :return: data_dict: same as input, with additional keys:
    - 'bb_target': bounding box coordinates (b, n_boxes, (y1, x1, y2, x2, (z1), (z2)))
    - 'roi_labels': corresponding class labels for each box (b, n_boxes, class_label)
    - 'roi_masks': corresponding binary segmentation mask for each lesion (box). Only used in Mask RCNN. (b, n_boxes, y, x, (z))
    - 'seg': now label map (see class_specific_seg_flag)
    '''

    bb_target = []
    roi_masks = []
    roi_labels = []
    out_seg = np.copy(data_dict['seg'])
    for b in range(data_dict['seg'].shape[0]):

        p_coords_list = []
        p_roi_masks_list = []
        p_roi_labels_list = []

        if np.sum(data_dict['seg'][b]!=0) > 0:
            if get_rois_from_seg_flag:
                clusters, n_cands = lb(data_dict['seg'][b])
                data_dict['class_target'][b] = [data_dict['class_target'][b]] * n_cands
            else:
                n_cands = int(np.max(data_dict['seg'][b]))
                clusters = data_dict['seg'][b]

            rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])  # separate clusters and concat
            for rix, r in enumerate(rois):
                if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
                    seg_ixs = np.argwhere(r != 0)
                    coord_list = [np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                     np.max(seg_ixs[:, 2])+1]
                    if dim == 3:

                        coord_list.extend([np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1])

                    p_coords_list.append(coord_list)
                    p_roi_masks_list.append(r)
                    # add background class = 0. rix is a patient wide index of lesions. since 'class_target' is
                    # also patient wide, this assignment is not dependent on patch occurrances.
                    p_roi_labels_list.append(data_dict['class_target'][b][rix] + 1)

                if class_specific_seg_flag:
                    out_seg[b][data_dict['seg'][b] == rix + 1] = data_dict['class_target'][b][rix] + 1

            if not class_specific_seg_flag:
                out_seg[b][data_dict['seg'][b] > 0] = 1

            bb_target.append(np.array(p_coords_list))
            roi_masks.append(np.array(p_roi_masks_list).astype('uint8'))
            roi_labels.append(np.array(p_roi_labels_list))


        else:
            bb_target.append([])
            roi_masks.append(np.zeros_like(data_dict['seg'][b])[None])
            roi_labels.append(np.array([-1]))

    if get_rois_from_seg_flag:
        data_dict.pop('class_target', None)

    data_dict['bb_target'] = np.array(bb_target)
    data_dict['roi_masks'] = np.array(roi_masks)
    data_dict['roi_labels'] = np.array(roi_labels)
    data_dict['seg'] = out_seg

    return data_dict

def bb_intersection_over_union(boxA, boxB):
    '''
    This function calculates the bounding box intersection.
    :param boxA: First bounding box in format [xmin,ymin,zmin,xmax,ymax,zmax]
    :param boxB: Second bounding box in format [xmin,ymin,zmin,xmax,ymax,zmax]
    :return: iou: Intersection over Union
    '''
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    zA = max(boxA[4], boxB[4])
    zB = min(boxA[5], boxB[5])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1) * max(0, zB - zA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1) * (boxA[5] - boxA[4] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1) * (boxB[5] - boxB[4] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def test_model(model, dataloaders):
    '''
    This function obtains the model predictions.
    :param model: ACME model
    :param boxB: dataloaders
    :return: score_clas: scores for classification
    results: array of bounding box predictions with scores
    '''
    since = time.time()
    
    # Eval mode
    model.eval()
    phase='test'
    model.apply(set_bn_eval)
    
    # Initialize variables
    running_corrects = 0
    running_frames = 0
    running_corrects_venule = 0
    running_frames_venule = 0
    batch_counter = 0
    scores = []
    true_label = []
    pred_label = []
    results=np.zeros((0,13),dtype='float')
    imgs_name='volumes'
    segms_name='soft'
    venules_name='venules'
    with torch.set_grad_enabled(False):   
        # Iterate over data.
        for inputs in dataloaders[phase]:
            # Obtain capture id
            paths=inputs[0]['path']
            aux=paths.replace(imgs_name,imgs_name+'_pred').split('/')
            aux2=aux[-1].split('__')
            grupo=int(aux2[0][-1])
            aux=aux2[1].split('_')
            ids=int(aux[0])
            capture=int(aux[1][7:])
            tt=int(aux2[2][1:])
            vol=int(aux2[3].split('.')[0])
            ids=np.array([grupo,ids,capture,tt,vol])
            initial_time = time.time() 
                
            results_dict= model.test_forward(inputs[0])               

            # statistics for cell segmentation
            labels=inputs[0]['seg'][:,0,:,:,:]
            prob=results_dict['seg_logits']
            preds=results_dict['seg_preds'][:,0,:,:,:]
            for i in range(preds.shape[0]):
                for j in range(preds.shape[3]):
                    predX=preds[i,:,:,j]
                    labelsX=labels[i,:,:,j]
                    if (np.sum(predX)>0 or np.sum(labelsX)>0):
                        running_frames+=1
                        running_corrects +=np.sum(np.logical_and(predX,labelsX))/(np.sum(labelsX)+np.sum(predX)-np.sum(np.logical_and(predX,labelsX)))
            
            # statistics for venule segmentation
            labels_venule=inputs[0]['venule'][:,0,:,:,:]
            prob_venule=results_dict['venule_logits']
            preds_venule=results_dict['venule_preds'][:,0,:,:,:]
            for i in range(preds_venule.shape[0]):
                for j in range(preds_venule.shape[3]):
                    predX=preds_venule[i,:,:,j]
                    labelsX=labels_venule[i,:,:,j]
                    if (np.sum(predX)>0 or np.sum(labelsX)>0):
                        running_frames_venule+=1
                        running_corrects_venule +=np.sum(np.logical_and(predX,labelsX))/(np.sum(labelsX)+np.sum(predX)-np.sum(np.logical_and(predX,labelsX)))
                        
            # classifier results
            bboxes2=np.squeeze(inputs[0]['bb_target'])
            if (len(bboxes2.shape)<2 and bboxes2.shape[0]==6):
                bboxes2=bboxes2.reshape(1,-1)
            bboxes1=np.zeros((0,6),dtype='float32')
            for i in range(len(results_dict['boxes'][0])):
                if (results_dict['boxes'][0][i]['box_type']=='det'):
                    scores.append(results_dict['boxes'][0][i]['box_score'])
                    pred_label.append(results_dict['boxes'][0][i]['box_pred_class_id'])
                    if (bboxes2.shape[0]==0):
                        true_label.append(0)
                        results=np.concatenate((results,np.array(np.concatenate((results_dict['boxes'][0][i]['box_coords'],np.array([results_dict['boxes'][0][i]['box_score']]),np.array([0]), ids),axis=0))[np.newaxis,:]), axis=0)
                    else:
                        reg=np.array(results_dict['boxes'][0][i]['box_coords']).astype('int')
                        reg[reg<0]=0
                        pred=preds[0,reg[0]:reg[2],reg[1]:reg[3],reg[4]:reg[5]]
                        gt=labels[0,reg[0]:reg[2],reg[1]:reg[3],reg[4]:reg[5]]
                        if (np.sum(pred)+np.sum(gt)-np.sum(pred*gt))>0:
                            overlap=np.sum(pred*gt)/(np.sum(pred)+np.sum(gt)-np.sum(pred*gt))
                        else:
                            overlap=0
                        if (overlap>0.5):
                            true_label.append(1)
                            results=np.concatenate((results,np.array(np.concatenate((results_dict['boxes'][0][i]['box_coords'],np.array([results_dict['boxes'][0][i]['box_score']]),np.array([1]), ids),axis=0))[np.newaxis,:]), axis=0)
                        else:
                            true_label.append(0)
                            results=np.concatenate((results,np.array(np.concatenate((results_dict['boxes'][0][i]['box_coords'],np.array([results_dict['boxes'][0][i]['box_score']]),np.array([0]), ids),axis=0))[np.newaxis,:]), axis=0)

                    bboxes1=np.concatenate((bboxes1,results_dict['boxes'][0][i]['box_coords'].reshape(1,-1)),axis=0)
            
            batch_counter += 1
            final_time = time.time() 
            ex_time = final_time - initial_time
            print('Batch {}/{} : {:.4f} s'.format(batch_counter,len(dataloaders[phase].dataset),ex_time))
            
            aux=paths.replace(imgs_name,imgs_name+'_pred').split(os.path.sep)
                
            # Paths for saving results
            img_pred_path=''
            for i in range(1,len(aux)-1):
                img_pred_path=img_pred_path+os.path.sep+aux[i]
            segm_pred_path=img_pred_path.replace(imgs_name+'_pred',segms_name+'_pred')
            venule_pred_path=img_pred_path.replace(imgs_name+'_pred',venules_name+'_pred')

            # Soft cell segmentations
            if not os.path.exists(segm_pred_path):
                os.makedirs(segm_pred_path)
            # Soft venule segmentations
            if not os.path.exists(venule_pred_path):
                os.makedirs(venule_pred_path)  
                
            segm_pred_path=segm_pred_path+os.path.sep+aux[-1]
            venule_pred_path=venule_pred_path+os.path.sep+aux[-1]

            segms_cpu=np.transpose(np.squeeze(prob),[1,2,3,0])
            sio.savemat(segm_pred_path[:-4]+'_pred.mat',{'segm':segms_cpu})
              
            venules_cpu=np.transpose(np.squeeze(prob_venule),[1,2,3,0])
            sio.savemat(venule_pred_path[:-4]+'_pred.mat',{'segm':venules_cpu})          


    if (running_frames>0):
        epoch_acc = running_corrects/running_frames
    else:
        epoch_acc = running_corrects
    if (running_frames_venule>0):
        epoch_acc_venule = running_corrects_venule/running_frames_venule
    else:
        epoch_acc_venule = running_corrects_venule
    if (len(np.unique(np.array(true_label)))>1):
        score_clas=roc_auc_score(true_label, scores)
        score_clas1=1.0
        score_clas2=1.0
    else:
        score_clas=1.0
        score_clas1=1.0
        score_clas2=1.0

    print('{} Acc: {:.4f} VenuleAcc: {:.4f}'.format(phase, epoch_acc, epoch_acc_venule))
    print('{} Auc Full: {:.4f} Auc1: {:.4f} Auc2: {:.4f}'.format(phase, score_clas,score_clas1, score_clas2))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return score_clas, results

class CNICDataset3d(object):
    '''
    This class implements data reading.
    :param root: the dataset root directory
    :param mean: RGB mean for 3D captures
    :param std: RGB std for 3D captures
    :param depth: depth of 3D volumes

    :return: score_clas: scores for classification
    results: array of bounding box predictions with scores
    '''
    def __init__(self, root, mean, std, depth):
        self.depth=depth
        self.mean = mean
        self.std = std
        self.root = root
        
        self.imgs=natsort.natsorted([os.path.join(root,name)
             for root, dirs, files in os.walk(root)
             for name in files if ('volumes' in root) and not ('volumes_init' in root)
             if (name.endswith(".mat") and not name.endswith("_pred.mat"))])

    def __getitem__(self, idx):
        # Load images ad masks
        img_path = os.path.join(self.imgs[idx])
        img = sio.loadmat(img_path)['data']
        img=img[None,:,:,:,:]#0:128:,0:128,0:8]
        masks=np.zeros((1,img.shape[0],img.shape[1],img.shape[2],img.shape[4]),dtype=np.uint8)
        venule=np.zeros((1,img.shape[0],img.shape[1],img.shape[2],img.shape[4]),dtype=np.uint8)
        # Convert to input format, with RoIs
        batch_3D={'data': img, 'seg': masks, 'venule': venule, 'pid': [0], 'class_target': [0]}
        converter = ConvertSegToBoundingBoxCoordinates(dim=3, get_rois_from_seg_flag=True, class_specific_seg_flag=False)
        batch_3D = converter(**batch_3D)
        masks=masks[0,0,:,:,:]
        venule=venule[0,0,:,:,:]
        img=img[0,:,:,:,:]
        masks = torch.as_tensor(np.transpose(masks,(2,0,1)).copy(), dtype=torch.float) #torch.squeeze(
        venule = torch.as_tensor(np.transpose(venule,(2,0,1)).copy(), dtype=torch.long) #torch.squeeze(

        # Normalize 
        img=torch.Tensor(np.transpose(img,(3,2,0,1)).copy())
        for j in range(img.size(0)):
            img[j,:,:,:]=F.normalize(img[j,:,:,:],torch.from_numpy(self.mean),torch.from_numpy(self.std))
        img=img.permute(1,2,3,0)[None,:,:,:,:,]

        if (masks.max()>1):
            masks[masks>1] = 1
        masks=masks.permute(1,2,0)[None,None,:,:,:,]
        venule=venule.permute(1,2,0)[None,None,:,:,:,]
        batch_3D['data']=img.numpy()
        batch_3D['seg']=masks.numpy()
        batch_3D['venule']=venule.numpy()
        batch_3D.update({'path': img_path})
        return batch_3D
    
    def __len__(self):
        return len(self.imgs)
    
def save_checkpoint(state, filename='models/checkpoint.pth.tar'):
    torch.save(state, filename)

def set_bn_eval(mm):
    if isinstance(mm, nn.modules.batchnorm._BatchNorm):
        mm.eval()

def collate_fn(batch):
    return batch


def main():
    
    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
         
    model_3d = ACMENet(cfg)
    
    # Send the model to GPU
    model_3d = model_3d.to(device)
    weights=torch.load(os.path.join(cfg.model_dir,cfg.model_name+'.pth'))['state_dict']
    
    model_3d.load_state_dict(weights)
    print("Initializing Datasets and Dataloaders for 3D...")
    
    # use our dataset and defined transformations
    dataset = CNICDataset3d(cfg.data_dir, cfg.mean, cfg.std, cfg.depth)
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size_3d, shuffle=False, num_workers=0,
        collate_fn=collate_fn)
    
    dataloaders_dict={'test': data_loader}
    
    [score,results]=test_model(model_3d, dataloaders_dict)
    sio.savemat(os.path.join(cfg.data_dir,'results_test.mat'),{'results':results})

if __name__ == "__main__":
    main()