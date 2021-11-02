#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:58:21 2021

@author: mmolina
"""

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

import model_utils as mutils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from scipy.ndimage.measurements import label as lb
from custom_extensions.roi_align import roi_align

############################################################
#  Network definition
############################################################

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

def conv3d(in_channels, out_channels, kernel_size, bias, padding=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, kernel_size=3, order='crg', num_groups=8):
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, order, num_groups))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, order, num_groups))
        
        
class AuxConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crg', num_groups=8):
        super(AuxConv, self).__init__()

        # conv1
        self.add_module('AuxSingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups,padding=0))

        # in the last layer a 1×1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.add_module('aux_final_conv', final_conv)
        
        
class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2, 2, 2), pool_type='max', basic_module=DoubleConv, conv_layer_order='crg',
                 num_groups=8):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 scale_factor=(2, 2, 2), basic_module=DoubleConv, conv_layer_order='crg', num_groups=8):
        super(Decoder, self).__init__()
        if basic_module == DoubleConv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining
            # works correctly
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               out_channels,
                                               kernel_size=kernel_size,
                                               stride=scale_factor,
                                               padding=1,
                                               output_padding=1)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and concatenation joining
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
            # concatenate encoder_features (encoder path) with the upsampled input across channel dimension
            x = torch.cat((encoder_features, x), dim=1)
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
            x += encoder_features

        x = self.basic_module(x)
        return x

class UNet3DAux(nn.Module):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, cf, conv, in_channels, out_channels, venule_channels, final_sigmoid, f_maps=32, layer_order='gcr', num_groups=8,
                 **kwargs):
        super(UNet3DAux, self).__init__()
        self.cf=cf
        self.conv=conv
        # Set testing mode to false by default. It has to be set to true in test mode, otherwise the `final_activation`
        # layer won't be applied
        self.testing = kwargs.get('testing', False)

        if isinstance(f_maps, int):
            # use 5 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=5)

        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(f_maps)`
        # uses DoubleConv as a basic_module for the Encoder
        encoders = []
        for i, out_feature_num in enumerate(f_maps):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, apply_pooling=False, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            elif i==1:
                encoder = Encoder(f_maps[i - 1], out_feature_num, pool_kernel_size=(2, 2, 1), basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                encoder = Encoder(f_maps[i - 1], out_feature_num, basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        pconv1s = []
        pconv2s = []
        # Add last encoders output convolution
        pconv1s.append(conv(out_feature_num, self.cf.end_filts, ks=1, stride=1, relu=None))
        pconv2s.append(conv(self.cf.end_filts, self.cf.end_filts, ks=3, stride=1, pad=1, relu=None))

        # create decoder path consisting of the Decoder modules. The length of the decoder is equal to `len(f_maps) - 1`
        # uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps) - 1):
            in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
            out_feature_num = reversed_f_maps[i + 1]
            if i==(len(reversed_f_maps) - 2):
                decoder = Decoder(in_feature_num, out_feature_num, scale_factor=(2,2,1), basic_module=DoubleConv,
                                  conv_layer_order=layer_order, num_groups=num_groups)
            else:
                decoder = Decoder(in_feature_num, out_feature_num, basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            if (i<self.cf.pyramid_levels[-1]):
                pconv1s.append(conv(out_feature_num, self.cf.end_filts, ks=1, stride=1, relu=None))
                pconv2s.append(conv(self.cf.end_filts, self.cf.end_filts, ks=3, stride=1, pad=1, relu=None))
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)
        
        self.pconv1s=nn.ModuleList(pconv1s)
        self.pconv2s=nn.ModuleList(pconv2s)

        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        self.venule_conv = nn.Conv3d(f_maps[0], venule_channels, 1)

        # aux classifier
        self.aux_final_conv=AuxConv(512, out_channels, 1, layer_order, num_groups)

    def forward(self, x):
        c = x.size()[2]
        h = x.size()[3]
        w = x.size()[4]
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
        f_maps=[]
        f_maps.append(self.pconv2s[0](self.pconv1s[0](x)))

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]
        y = F.interpolate(self.aux_final_conv(x), size=(c, h, w), mode='trilinear', align_corners=True)

        k=0
        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            if (k<max(self.cf.pyramid_levels)):
                f_maps.append(self.pconv2s[k+1](self.pconv1s[k+1](x)))
            k=k+1
            
        v = self.venule_conv(x)
        x = self.final_conv(x)

        f_maps.insert(0,x)

        return {'f_maps': f_maps, 'aux': y, 'venule': v}


############################################################
#  Loss Functions
############################################################
        
import torch.nn.functional as FT

class FocalLossBCE(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = (ln(MSELoss)/ln(max(MSELoss)))^gamma MSELoss
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, gamma=2, alpha=1, size_average=True):
        super(FocalLossBCE, self).__init__()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        
        BCE_loss = FT.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # prevents nans when probability 0
        batch_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

def compute_segmentation_classifier_loss(target_class_ids, pred_class_logits):
    """
    :param target_class_ids: (n_sampled_rois) batch dimension was merged into roi dimension.
    :param pred_class_logits: (n_sampled_rois, n_classes)
    :return: loss: torch 1D tensor.
    """
    if 0 not in target_class_ids.size():
        loss = F.cross_entropy(pred_class_logits, target_class_ids.long(),ignore_index=-1)
    else:
        loss = torch.FloatTensor([0.]).cuda()

    return loss

def compute_class_loss(anchor_matches, class_pred_logits, shem_poolsize=20):
    """
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :param class_pred_logits: (n_anchors, n_classes). logits from classifier sub-network.
    :param shem_poolsize: int. factor of top-k candidates to draw from per negative sample (online-hard-example-mining).
    :return: loss: torch tensor.
    :return: np_neg_ix: 1D array containing indices of the neg_roi_logits, which have been sampled for training.
    """
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    pos_indices = torch.nonzero(anchor_matches > 0)
    neg_indices = torch.nonzero(anchor_matches == -1)

    # get positive samples and calucalte loss.
    if 0 not in pos_indices.size():
        pos_indices = pos_indices.squeeze(1)
        roi_logits_pos = class_pred_logits[pos_indices]
        targets_pos = anchor_matches[pos_indices]
        pos_loss = F.cross_entropy(roi_logits_pos, targets_pos.long())
    else:
        pos_loss = torch.FloatTensor([0]).cuda()

    # get negative samples, such that the amount matches the number of positive samples, but at least 1.
    # get high scoring negatives by applying online-hard-example-mining.
    if 0 not in neg_indices.size():
        neg_indices = neg_indices.squeeze(1)
        roi_logits_neg = class_pred_logits[neg_indices]
        negative_count = np.max((1, pos_indices.size()[0]))
        roi_probs_neg = F.softmax(roi_logits_neg, dim=1)
        neg_ix = mutils.shem(roi_probs_neg, negative_count, shem_poolsize)
        neg_loss = F.cross_entropy(roi_logits_neg[neg_ix], torch.LongTensor([0] * neg_ix.shape[0]).cuda())
        # return the indices of negative samples, which contributed to the loss (for monitoring plots).
        np_neg_ix = neg_ix.cpu().data.numpy()
    else:
        neg_loss = torch.FloatTensor([0]).cuda()
        np_neg_ix = np.array([]).astype('int32')

    loss = (pos_loss + neg_loss) / 2
    return loss, np_neg_ix


def compute_bbox_loss(target_deltas, pred_deltas, anchor_matches):
    """
    :param target_deltas:   (b, n_positive_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd)))).
    Uses 0 padding to fill in unsed bbox deltas.
    :param pred_deltas: predicted deltas from bbox regression head. (b, n_anchors, (dy, dx, (dz), log(dh), log(dw), (log(dd))))
    :param anchor_matches: (n_anchors). [-1, 0, class_id] for negative, neutral, and positive matched anchors.
    :return: loss: torch 1D tensor.
    """
    if 0 not in torch.nonzero(anchor_matches > 0).size():

        indices = torch.nonzero(anchor_matches > 0).squeeze(1)
        # Pick bbox deltas that contribute to the loss
        pred_deltas = pred_deltas[indices]
        # Trim target bounding box deltas to the same length as pred_deltas.
        target_deltas = target_deltas[:pred_deltas.size()[0], :]
        # Smooth L1 loss
        loss = F.smooth_l1_loss(pred_deltas, target_deltas)
    else:
        loss = torch.FloatTensor([0]).cuda()

    return loss

############################################################
#  Inference
############################################################

def get_results_classifier(cf, img_shape, detections, seg_logits, venule_logits, box_results_list=None):
    """
    Restores batch dimension of merged detections, unmolds detections, creates and fills results dict.
    :param img_shape:
    :param detections: (n_final_detections, (y1, x1, y2, x2, (z1), (z2), batch_ix, pred_class_id, pred_score)
    :param box_results_list: None or list of output boxes for monitoring/plotting.
    each element is a list of boxes per batch element.
    :return: results_dict: dictionary with keys:
             'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                      [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
             'seg_preds_cell': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, 1] for cell segmentation.
             'seg_preds_venule': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, 1, 2] for background-venule-cell segmentation.
             'seg_logits_cell': pixel-wise logits (b, 1, y, x, (z)) for cell segmentation.
             'seg_logits_venule': pixel-wise logits (b, 1, y, x, (z)) for background-venule-cell segmentation.

    """
    if (detections.size(0)==0):
        if box_results_list is None:
            box_results_list = [[] for _ in range(img_shape[0])] 

    else:
        detections = detections.cpu().data.numpy()
        batch_ixs = detections[:, cf.dim*2]
        detections = [detections[batch_ixs == ix] for ix in range(img_shape[0])]
    
        # for test_forward, where no previous list exists.
        if box_results_list is None:
            box_results_list = [[] for _ in range(img_shape[0])]
    
        for ix in range(img_shape[0]):
    
            if 0 not in detections[ix].shape:
    
                boxes = detections[ix][:, :2 * cf.dim].astype(np.int32)
                class_ids = detections[ix][:, 2 * cf.dim + 1].astype(np.int32)
                scores = detections[ix][:, 2 * cf.dim + 2]
                if (detections[ix].shape[1] == 2*cf.dim+4):
                    target_ids = detections[ix][:, 2 * cf.dim + 3].astype(np.int32)
                    
    
    
                # Filter out detections with zero area. Often only happens in early
                # stages of training when the network weights are still a bit random.
                if cf.dim == 2:
                    exclude_ix = np.where((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
                else:
                    exclude_ix = np.where(
                        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 5] - boxes[:, 4]) <= 0)[0]
    
                if exclude_ix.shape[0] > 0:
                    boxes = np.delete(boxes, exclude_ix, axis=0)
                    class_ids = np.delete(class_ids, exclude_ix, axis=0)
                    scores = np.delete(scores, exclude_ix, axis=0)
                    if (detections[ix].shape[1] == 2*cf.dim+4):
                        target_ids = np.delete(target_ids, exclude_ix, axis=0)
    
                if 0 not in boxes.shape:
                    for ix2, score in enumerate(scores):
                        if score >= cf.model_min_confidence:
                            if (detections[ix].shape[1] == 2*cf.dim+4):
                                box_results_list[ix].append({'box_coords': boxes[ix2],
                                                             'box_score': score,
                                                             'box_type': 'det',
                                                             'box_pred_class_id': class_ids[ix2],
                                                             'box_true_class_id': target_ids[ix2]})
                            else:
                                box_results_list[ix].append({'box_coords': boxes[ix2],
                                                             'box_score': score,
                                                             'box_type': 'det',
                                                             'box_pred_class_id': class_ids[ix2]})
    
    results_dict = {'boxes': box_results_list}
    if seg_logits is None:
        # output dummy segmentation for retina_net.
        results_dict['seg_preds'] = np.zeros(img_shape)[:, 0][:, np.newaxis]
        results_dict['venule_preds'] = np.zeros(img_shape)[:, 0][:, np.newaxis]
        results_dict['seg_logits'] = np.concatenate((np.ones(img_shape)[:, 0][:, np.newaxis],np.zeros(img_shape)[:, 0][:, np.newaxis]),axis=1)
        results_dict['venule_logits'] = np.concatenate((np.ones(img_shape)[:, 0][:, np.newaxis],np.zeros(img_shape)[:, 0][:, np.newaxis],np.zeros(img_shape)[:, 0][:, np.newaxis]),axis=1)
    else:
        # output label maps for retina_unet.
        results_dict['seg_preds'] = F.softmax(seg_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis].astype('uint8')
        results_dict['venule_preds'] = F.softmax(venule_logits, 1).argmax(1).cpu().data.numpy()[:, np.newaxis]
        results_dict['venule_preds'] = (results_dict['venule_preds']>0).astype('uint8')
        results_dict['seg_logits'] = seg_logits.cpu().data.numpy()
        results_dict['venule_logits'] = venule_logits.cpu().data.numpy()

    return results_dict

############################################################
#  Classification Branch
############################################################

def pyramid_roi_align(feature_maps, rois, pool_size, pyramid_levels, dim):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    :param feature_maps: list of feature maps, each of shape (b, c, y, x , (z))
    :param rois: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
    (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ixs)
    :param pool_size: list of poolsizes in dims: [x, y, (z)]
    :param pyramid_levels: list. [0, 1, 2, ...]
    :return: pooled: pooled feature map rois (n_proposals, c, poolsize_y, poolsize_x, (poolsize_z))

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    boxes = rois[:, :dim*2]
    batch_ixs = rois[:, dim*2]

    # Assign each ROI to a level in the pyramid based on the ROI area.
    if dim == 2:
        y1, x1, y2, x2 = boxes.chunk(4, dim=1)
    else:
        y1, x1, y2, x2, z1, z2 = boxes.chunk(6, dim=1)

    h = y2 - y1
    w = x2 - x1

    # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
    # the fact that our coordinates are normalized here.
    # divide sqrt(h*w) by 1 instead image_area.
    roi_level = (4 + torch.log2(torch.sqrt(h*w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])
    # if Pyramid contains additional level P6, adapt the roi_level assignment accordingly.
    if len(pyramid_levels) == 5:
        roi_level[h*w > 0.65] = 5

    # Loop through levels and apply ROI pooling to each.
    pooled = []
    box_to_level = []
    fmap_shapes = [f.shape for f in feature_maps]
    for level_ix, level in enumerate(pyramid_levels):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix, :]
        # re-assign rois to feature map of original batch element.
        ind = batch_ixs[ix].int()

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()
        if len(pool_size) == 2:
            # remap to feature map coordinate system
            y_exp, x_exp = fmap_shapes[level_ix][2:]  # exp = expansion
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_2d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        else:
            y_exp, x_exp, z_exp = fmap_shapes[level_ix][2:]
            level_boxes.mul_(torch.tensor([y_exp, x_exp, y_exp, x_exp, z_exp, z_exp], dtype=torch.float32).cuda())
            pooled_features = roi_align.roi_align_3d(feature_maps[level_ix],
                                                     torch.cat((ind.unsqueeze(1).float(), level_boxes), dim=1),
                                                     pool_size)
        pooled.append(pooled_features)
        del pooled_features, level_boxes, ix

    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled
    
class Well_Segmented_Cell_Classifier(nn.Module):
    """
    Head network for classification of cells. Performs RoiAlign, processes resulting features through a
    shared convolutional base and finally provides the well-segmented cell segmentation probability.
    """
    def __init__(self, cf, conv):
        super(Well_Segmented_Cell_Classifier, self).__init__()

        self.dim = conv.dim
        self.in_channels = cf.end_filts
        self.pool_size = cf.pool_size# (7,7,3) mask_pool_size  (14,14,5)
        self.pyramid_levels = cf.pyramid_levels
        # instance_norm does not work with spatial dims (1, 1, (1))
        norm = cf.norm if cf.norm != 'instance_norm' else None

        self.conv1 = conv(cf.end_filts*3, cf.end_filts*8, ks=self.pool_size, stride=1, norm=norm, relu=cf.relu)
        self.conv2 = conv(cf.end_filts*8, cf.end_filts*8, ks=1, stride=1, norm=norm, relu=cf.relu)
        self.linear_class = nn.Linear(cf.end_filts*8, cf.head_classes)
        
    def forward(self, x, rois):
        """
        :param x: input feature maps (b, in_channels, y, x, (z))
        :param rois: normalized box coordinates as proposed by the RPN to be forwarded through
        the second stage (n_proposals, (y1, x1, y2, x2, (z1), (z2), batch_ix). Proposals of all batch elements
        have been merged to one vector, while the origin info has been stored for re-allocation.
        :return: segmentation_classifier_logits (n_proposals, n_head_classes)
        """

        x = pyramid_roi_align(x, rois, self.pool_size, self.pyramid_levels, self.dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.in_channels * 8)
        segmentation_classifier_logits = self.linear_class(x)

        return segmentation_classifier_logits

############################################################
#  3D Joint Segmentation Module Class
############################################################


class net(nn.Module):


    def __init__(self, cf):

        super(net, self).__init__()
        self.cf = cf
        self.build()

    def build(self):
        """
        Build Retina Net architecture.
        """

        # Image size must be dividable by 2 multiple times.
        h, w = self.cf.patch_size[:2]
        if h / 2 ** 5 != int(h / 2 ** 5) or w / 2 ** 5 != int(w / 2 ** 5):
            raise Exception("Image size must be dividable by 2 at least 5 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # instanciate abstract multi dimensional conv class and backbone model.
        conv = mutils.NDConvGenerator(self.cf.dim)
        
        # U-Net 3D Aux
        self.Unet = UNet3DAux(self.cf, conv, in_channels=3, out_channels=2, venule_channels=3, final_sigmoid=False)
        for i in range(len(self.Unet.pconv1s)):
            self.Unet.pconv1s[i]=conv(self.Unet.pconv1s[i].in_channels, self.cf.end_filts*3, ks=self.Unet.pconv1s[i].kernel_size, stride=self.Unet.pconv1s[i].stride, relu=None)
        for i in range(len(self.Unet.pconv2s)):
            self.Unet.pconv2s[i]=conv(self.cf.end_filts*3, self.cf.end_filts*3, ks=self.Unet.pconv2s[i].kernel_size, stride=self.Unet.pconv2s[i].stride, pad=self.Unet.pconv2s[i].padding, relu=None)
        # Classifier
        self.wscc = Well_Segmented_Cell_Classifier(self.cf, conv)

    def test_forward(self, batch, **kwargs):
        """
        test method. wrapper around forward pass of network without usage of any ground truth information.
        prepares input data for processing and stores outputs in a dictionary.
        :param batch: dictionary containing 'data'
        :return: results_dict: dictionary with keys:
               'boxes': list over batch elements. each batch element is a list of boxes. each box is a dictionary:
                       [[{box_0}, ... {box_n}], [{box_0}, ... {box_n}], ...]
               'seg_preds...': pixel-wise class predictions (b, 1, y, x, (z)) with values [0, ..., n_classes] 
               for cell and venule segmentations.
        """
        img = batch['data']
        img = torch.from_numpy(img).float().cuda()
        outputs = self.forward(img)
        fpn_outs = outputs['f_maps']
        seg_logits = fpn_outs[0]
        venule_logits = outputs['venule']

        prob = FT.softmax(seg_logits,dim=1)
        _, preds = torch.max(prob, 1)  
        preds=preds.squeeze().cpu().numpy()
        
        # Region generation
        clusters, n_cands =lb(preds[np.newaxis,:,:,:])
        rois = np.array([(clusters == ii) * 1 for ii in range(1, n_cands + 1)])
        segmented_rois=np.zeros((n_cands,7),dtype='float32')
        for rix, r in enumerate(rois):
            if np.sum(r !=0) > 0: #check if the lesion survived data augmentation
                seg_ixs = np.argwhere(r != 0)
                segmented_rois[rix,:] = np.array([np.min(seg_ixs[:, 1])-1, np.min(seg_ixs[:, 2])-1, np.max(seg_ixs[:, 1])+1,
                                  np.max(seg_ixs[:, 2])+1, np.min(seg_ixs[:, 3])-1, np.max(seg_ixs[:, 3])+1, 0])
            
            
        # Normalize regions
        segmented_rois=segmented_rois/np.concatenate((np.repeat(np.array(self.cf.patch_size),2),np.array([1])),axis=0)
        segmented_rois=torch.as_tensor(segmented_rois,dtype=torch.float32).cuda()
        # Classify the regions (obtain its well-segmented probability)
        if (segmented_rois.size(0)>0):
            segmentation_classifier_logits = self.wscc(fpn_outs[1:], segmented_rois)
        else:
            segmentation_classifier_logits = torch.zeros((segmented_rois.shape[0],2),dtype=torch.float32).cuda()
        
        # Unnormalize the detections
        if (segmented_rois.size(0)>0):
            score=F.softmax(segmentation_classifier_logits, 1);
            _,segmented_pred=torch.max(score,dim=1)
            segmented_rois=segmented_rois*torch.as_tensor(np.concatenate((np.repeat(np.array(self.cf.patch_size),2),np.array([1])),axis=0)).type(torch.float32).cuda()
            detections=torch.cat((segmented_rois,segmented_pred.type(torch.float32)[:,None],score[:,1][:,None]),dim=1)
        else:
            detections=torch.zeros((segmented_rois.shape[0],9),dtype=torch.float32).cuda()
        
        # Obtain the result dictionary
        results_dict = get_results_classifier(self.cf, img.shape, detections, seg_logits, venule_logits)
        return results_dict


    def forward(self, img):
        """
        forward pass of the model.
        :param img: input img (b, c, y, x, (z)).
        :return: fpn_outs: venule and cell segmentations and feature maps for classifier 
        """
        # Feature extraction
        fpn_outs = self.Unet(img)
        return fpn_outs
