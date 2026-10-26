#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for the ACME 3D Joint Segmentation Module
(Joint3DSegmentationModule.net: U-Net 3D with auxiliary and venule heads +
well-segmented cell classifier).

Place this file in ACME/python/, next to Joint3DSegmentationModule.py,
config.py and model_utils.py.

All settings are read from config.py (cfg.TRAIN for training, cfg.data_dir and
cfg.model_dir for the folders), so the file can be run directly (e.g. from
VSCode) without command-line arguments.

Two training stages are available (cfg.TRAIN.stage = 1, 2 or 'both'):
  * Stage 1: segmentation training (cell, auxiliary and venule losses; all
    parameters, single learning rate, StepLR). Optionally initialised from
    cfg.TRAIN.init_weights.
  * Stage 2: fine-tuning with differential learning rates: low LR for the
    U-Net body (encoders, decoders, segmentation/venule/aux heads) and
    higher LR for the feature-pyramid convolutions and the classifier. The loss
    is the classifier loss + 0.001 * segmentation losses. With stage 'both' it
    starts from the best stage-1 checkpoint; with stage 2 alone, from
    cfg.TRAIN.init_weights if set, otherwise from that checkpoint.

Expected database layout (default: cfg.data_dir = ACME/database):
  <data_dir>/train/volumes/*.mat   ('data':   H x W x C x D)
  <data_dir>/train/labels/*.mat    ('label':  H x W x D, cells > 0)
  <data_dir>/train/venules/*.mat   ('venule': H x W x D, venule > 0)
  <data_dir>/test/...              (same structure, used for validation)

@author: mmolina
"""

import copy
import logging
import os
import random
import time

import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data
import torchvision.transforms.functional as TF
from scipy.ndimage import find_objects
from scipy.ndimage import label as cc_label
from sklearn.metrics import roc_auc_score

import model_utils as mutils
from config import cfg
from Joint3DSegmentationModule import net as ACMENet


############################################################
#  Utilities
############################################################

def get_logger(log_dir, name='acme_train'):
    """Logger writing to <log_dir>/exec.log and to the terminal."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')
    for hdlr in (logging.FileHandler(os.path.join(log_dir, 'exec.log')),
                 logging.StreamHandler()):
        hdlr.setFormatter(fmt)
        logger.addHandler(hdlr)
    logger.propagate = False
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_weights(model, path, logger):
    """Load a checkpoint ({'state_dict': ...} or a plain state dict).
    Non-strict, so partially matching checkpoints (e.g. a segmentation-only
    pretraining) can be used; mismatches are reported."""
    ckpt = torch.load(path, map_location='cpu')
    state = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    own = model.state_dict()
    compatible = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = [k for k in state if k not in compatible]
    missing = [k for k in own if k not in compatible]
    model.load_state_dict(compatible, strict=False)
    logger.info('Loaded {}: {} tensors loaded, {} skipped, {} not initialised'.format(
        path, len(compatible), len(skipped), len(missing)))
    if missing:
        logger.info('  Not initialised (first 10): {}'.format(missing[:10]))


def seg_to_boxes(seg):
    """Connected components of a binary (H, W, D) mask -> boxes (n, 6) in the
    medicaldetectiontoolkit format (y1, x1, y2, x2, z1, z2) with 1-voxel margin."""
    clusters, _ = cc_label(seg > 0)
    boxes = [[sy.start - 1, sx.start - 1, sy.stop, sx.stop, sz.start - 1, sz.stop]
             for sy, sx, sz in find_objects(clusters)]
    return np.array(boxes, dtype=np.float32).reshape(-1, 6)


def iou_2d_slices(pred, gt):
    """Accumulate per-slice IoU over depth for binary (H, W, D) volumes.
    Returns (sum of IoUs, number of non-empty slices)."""
    total, frames = 0.0, 0
    for j in range(pred.shape[-1]):
        p, g = pred[..., j] > 0, gt[..., j] > 0
        union = np.logical_or(p, g).sum()
        if union > 0:
            total += np.logical_and(p, g).sum() / union
            frames += 1
    return total, frames


def compute_channel_stats(data_dir, logger):
    """Per-channel mean/std over the training volumes (H x W x C x D)."""
    vols = list_volumes(data_dir, 'train')
    s, s2, n = 0.0, 0.0, 0
    for k, path in enumerate(vols):
        img = sio.loadmat(path)['data'].astype(np.float64)
        z = np.moveaxis(img, 2, 0).reshape(img.shape[2], -1)
        s = s + z.sum(1)
        s2 = s2 + (z ** 2).sum(1)
        n += z.shape[1]
        if k % 50 == 0:
            logger.info('Statistics: {}/{}'.format(k, len(vols)))
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    return mean.astype(np.float32), std.astype(np.float32)


############################################################
#  Dataset
############################################################

def list_volumes(root, split):
    folder = os.path.join(root, split, 'volumes')
    return sorted(os.path.join(dp, f) for dp, _, files in os.walk(folder) for f in files
                  if f.endswith('.mat') and not f.endswith('_pred.mat'))


class CNICDataset3d(torch.utils.data.Dataset):
    """
    :param root: database root, with train/ and test/ subfolders.
    :param mean, std: per-channel statistics (length C).
    :param split: 'train' or 'test'.
    :param augment: random intensity, flip and affine augmentation.
    """
    def __init__(self, root, mean, std, split='train', augment=False):
        self.mean = torch.as_tensor(np.asarray(mean, dtype=np.float32))
        self.std = torch.as_tensor(np.asarray(std, dtype=np.float32))
        self.augment = augment
        self.imgs = list_volumes(root, split)
        if len(self.imgs) == 0:
            raise RuntimeError('No volumes found in {}'.format(os.path.join(root, split, 'volumes')))

    @staticmethod
    def _label_path(img_path, folder):
        head, tail = os.path.split(img_path)
        return os.path.join(os.path.join(os.path.dirname(head), folder), tail)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = sio.loadmat(img_path)['data'].astype(np.float32)                  # H x W x C x D
        masks = sio.loadmat(self._label_path(img_path, 'labels'))['label']
        masks = (np.asarray(masks) > 0).astype(np.uint8)                          # H x W x D
        venule = np.asarray(sio.loadmat(self._label_path(img_path, 'venules'))['venule']).astype(np.uint8)

        if self.augment:
            # Gamma and brightness adjustment per channel
            for c in range(img.shape[2]):
                gamma = 0.5 * (random.random() - 0.5) + 1
                brightness = 0.3 * (random.random() - 0.5) + 1
                img[:, :, c, :] = np.clip(img[:, :, c, :], 0, None) ** gamma * brightness
            # Random flips
            for axis in (0, 1):
                if random.random() < 0.5:
                    img, masks, venule = (np.flip(img, axis), np.flip(masks, axis), np.flip(venule, axis))
            img, masks, venule = (np.ascontiguousarray(a) for a in (img, masks, venule))
            # Random affine (rotation, scale, translation)
            h, w = masks.shape[:2]
            rot_mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), random.randint(-10, 10),
                                              1 - 0.1 * (random.random() - 0.5))
            rot_mat[0, 2] += random.randint(-10, 10)
            rot_mat[1, 2] += random.randint(-10, 10)
            for d in range(img.shape[3]):
                for c in range(img.shape[2]):
                    img[:, :, c, d] = cv2.warpAffine(img[:, :, c, d], rot_mat, dsize=(w, h))
                masks[:, :, d] = cv2.warpAffine(masks[:, :, d], rot_mat, dsize=(w, h), flags=cv2.INTER_NEAREST)
                venule[:, :, d] = cv2.warpAffine(venule[:, :, d], rot_mat, dsize=(w, h), flags=cv2.INTER_NEAREST)

        # Normalisation per depth slice (same as detection_inference.py)
        img_t = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (3, 2, 0, 1))))  # D x C x H x W
        for d in range(img_t.size(0)):
            img_t[d] = TF.normalize(img_t[d], self.mean, self.std)
        img_t = img_t.permute(1, 2, 3, 0)[None]                                            # 1 x C x H x W x D

        boxes = seg_to_boxes(masks)
        return {'data': img_t.numpy(),
                'seg': masks[None, None],
                'venule': venule[None, None].astype(np.int64),
                'bb_target': boxes[None],                                          # 1 x n x 6
                'roi_labels': np.ones((1, boxes.shape[0]), dtype=np.int64),       # single class: cell
                'pid': [os.path.basename(img_path)],
                'path': img_path}

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return batch


############################################################
#  Training loop
############################################################

def run_epoch(model, loader, phase, optimizer, logger, epoch, num_epochs, log_every, stage):
    is_train = phase == 'train'
    model.train() if is_train else model.eval()

    running_loss = 0.0
    cell_iou, cell_frames, ven_iou, ven_frames = 0.0, 0, 0.0, 0
    ret, rel, ret_rel = 0, 0, 0
    scores, true_label = [], []
    n_batches = len(loader)

    for b_ix, inputs in enumerate(loader):
        t0 = time.time()
        batch = inputs[0]
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            results_dict = model.train_forward(batch, stage=stage)
            if is_train:
                results_dict['torch_loss'].backward()
                optimizer.step()
        running_loss += results_dict['torch_loss'].item()

        # Cell and venule segmentation (per-slice IoU)
        s, f = iou_2d_slices(results_dict['seg_preds'][0, 0], batch['seg'][0, 0])
        cell_iou, cell_frames = cell_iou + s, cell_frames + f
        s, f = iou_2d_slices(results_dict['venule_preds'][0, 0], batch['venule'][0, 0])
        ven_iou, ven_frames = ven_iou + s, ven_frames + f

        # Detection precision/recall (box IoU > 0.5 with ground-truth cells)
        gt_boxes = batch['bb_target'][0]
        det_boxes = np.array([bx['box_coords'] for bx in results_dict['boxes'][0]
                              if bx['box_type'] == 'det'], dtype=np.float32).reshape(-1, 6)
        rel += gt_boxes.shape[0]
        ret += det_boxes.shape[0]
        if gt_boxes.shape[0] > 0 and det_boxes.shape[0] > 0:
            ret_rel += int(np.sum(np.max(mutils.compute_overlaps(gt_boxes, det_boxes), axis=1) > 0.5))

        # Classifier (well-segmented cell) scores; regions in the ignore band (-1) are skipped
        for bx in results_dict['boxes'][0]:
            if bx['box_type'] == 'det' and bx.get('box_true_class_id', -1) >= 0:
                scores.append(float(bx['box_score']))
                true_label.append(int(bx['box_true_class_id']))

        if (b_ix + 1) % log_every == 0 or b_ix + 1 == n_batches:
            logger.info('[Epoch {}/{} {}] Batch {}/{} ({:.2f} s) {}'.format(
                epoch, num_epochs - 1, phase, b_ix + 1, n_batches, time.time() - t0,
                results_dict['logger_string']))

    metrics = {
        'loss': running_loss / max(n_batches, 1),
        'acc': cell_iou / cell_frames if cell_frames > 0 else 0.0,
        'venule_acc': ven_iou / ven_frames if ven_frames > 0 else 0.0,
        'precision': ret_rel / ret if ret > 0 else 0.0,
        'recall': ret_rel / rel if rel > 0 else 0.0,
        'auc': float(roc_auc_score(true_label, scores)) if len(np.unique(true_label)) > 1 else float('nan'),
    }
    logger.info('{} Loss: {loss:.4f} Acc: {acc:.4f} VenuleAcc: {venule_acc:.4f} Prec: {precision:.4f} '
                'Rec: {recall:.4f} Auc: {auc:.4f}'.format(phase, **metrics))
    return metrics


def train_model(model, dataloaders, optimizer, lr_scheduler, num_epochs, out_dir, run_name,
                logger, stage, save_every=0, log_every=10):
    """Trains and validates; keeps the best model by validation cell IoU
    ('acc') and by validation classifier AUC ('auc')."""
    since = time.time()
    history = {p: {k: [] for k in ('loss', 'acc', 'venule_acc', 'precision', 'recall', 'auc')}
               for p in ('train', 'val')}
    best = {'acc': (-1.0, copy.deepcopy(model.state_dict())),
            'auc': (-1.0, None)}
    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        for phase in ('train', 'val'):
            m = run_epoch(model, dataloaders[phase], phase, optimizer, logger, epoch, num_epochs, log_every, stage)
            for k, v in m.items():
                history[phase][k].append(v)
            if phase == 'val':
                for key in best:
                    if not np.isnan(m[key]) and m[key] > best[key][0]:
                        best[key] = (m[key], copy.deepcopy(model.state_dict()))
                        save_checkpoint({'state_dict': best[key][1], 'epoch': epoch, key: float(m[key])},
                                        os.path.join(out_dir, '{}-best_{}.pth'.format(run_name, key)))
                        logger.info('New best val {}: {:.4f}'.format(key, m[key]))

        last = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': lr_scheduler.state_dict(), 'epoch': epoch}
        save_checkpoint(last, os.path.join(out_dir, '{}-last.pth'.format(run_name)))
        if save_every > 0 and (epoch % save_every == 0 or epoch == num_epochs - 1):
            save_checkpoint(last, os.path.join(out_dir, '{}-epoch{}.pth'.format(run_name, epoch)))
        lr_scheduler.step()

    elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
    logger.info('Best val Acc: {:.4f}  Best val AUC: {:.4f}'.format(best['acc'][0], best['auc'][0]))

    if best['auc'][0] < 0:
        logger.info('Validation AUC was never defined (only one class among the regions); '
                    'no best_auc checkpoint for {}'.format(run_name))

    # Training histories (.mat)
    for phase in history:
        for k, v in history[phase].items():
            name = '{}_{}'.format(phase, k)
            sio.savemat(os.path.join(out_dir, '{}_{}.mat'.format(name, run_name)), {name: np.array(v)})

    model.load_state_dict(best['acc'][1])
    return model, history


############################################################
#  Optimisers for each stage
############################################################

def stage1_optimizer(model, args):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr1, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step1, gamma=args.gamma1)
    return optimizer, scheduler


def stage2_optimizer(model, args):
    """Low LR for the U-Net body, higher LR for pyramid convs and classifier."""
    head_modules = [model.Unet.pconv1s, model.Unet.pconv2s, model.wscc]
    head_ids = {id(p) for mod in head_modules for p in mod.parameters()}
    heads = [p for p in model.parameters() if id(p) in head_ids and p.requires_grad]
    body = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    optimizer = torch.optim.Adam([{'params': body, 'lr': args.lr2_body},
                                  {'params': heads, 'lr': args.lr2_heads}],
                                 betas=(0.9, 0.999), weight_decay=args.weight_decay, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step2, gamma=args.gamma2)
    return optimizer, scheduler


############################################################
#  Main
############################################################

def get_train_options():
    """Training options from config.py (cfg.TRAIN), plus the data and model folders."""
    if 'TRAIN' not in cfg:
        raise KeyError('config.py has no TRAIN section (cfg.TRAIN); see the training options block.')
    opts = copy.deepcopy(cfg.TRAIN)
    opts.stage = str(opts.stage)
    if opts.stage not in ('1', '2', 'both'):
        raise ValueError("cfg.TRAIN.stage must be 1, 2 or 'both', got {!r}".format(cfg.TRAIN.stage))
    if opts.stage2_from not in ('acc', 'auc'):
        raise ValueError("cfg.TRAIN.stage2_from must be 'acc' or 'auc'")
    opts.data_dir = cfg.data_dir
    opts.model_dir = cfg.model_dir
    return opts


def main():
    args = get_train_options()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required (Joint3DSegmentationModule and roi_align run on GPU).')
    set_seed(args.seed)

    out_dir = os.path.join(args.model_dir, args.model_name)
    logger = get_logger(out_dir)
    logger.info('Training options: {}'.format(dict(args)))

    # Statistics
    if args.recompute_stats:
        mean, std = compute_channel_stats(args.data_dir, logger)
        np.save(os.path.join(out_dir, 'mean-frame.npy'), mean)
        np.save(os.path.join(out_dir, 'std-frame.npy'), std)
        logger.info('Statistics saved in {} (use them for inference too)'.format(out_dir))
    else:
        mean, std = cfg.mean, cfg.std

    # Data (batch size 1 only, as in the rest of ACME)
    datasets = {'train': CNICDataset3d(args.data_dir, mean, std, 'train', augment=True),
                'val': CNICDataset3d(args.data_dir, mean, std, 'test', augment=False)}
    dataloaders = {k: torch.utils.data.DataLoader(d, batch_size=cfg.batch_size_3d, shuffle=(k == 'train'),
                                                  num_workers=args.num_workers, collate_fn=collate_fn)
                   for k, d in datasets.items()}
    logger.info('Training volumes: {}  Validation volumes: {}'.format(
        len(datasets['train']), len(datasets['val'])))

    model = ACMENet(cfg).cuda()

    stages = ['1', '2'] if args.stage == 'both' else [args.stage]

    if '1' in stages:
        logger.info('===== STAGE 1: segmentation training =====')
        if args.init_weights:
            load_weights(model, args.init_weights, logger)
        optimizer, scheduler = stage1_optimizer(model, args)
        model, _ = train_model(model, dataloaders, optimizer, scheduler, args.epochs1, out_dir,
                               args.model_name + '_stage1', logger, 1, args.save_every, args.log_every)

    if '2' in stages:
        logger.info('===== STAGE 2: fine-tuning =====')
        stage1_ckpt = os.path.join(out_dir, '{}_stage1-best_{}.pth'.format(args.model_name, args.stage2_from))
        if args.stage == 'both':
            init = stage1_ckpt
        else:
            init = args.init_weights or stage1_ckpt
        if not os.path.isfile(init):
            raise FileNotFoundError('Stage 2 needs initial weights; not found: {}'.format(init))
        load_weights(model, init, logger)
        optimizer, scheduler = stage2_optimizer(model, args)
        model, _ = train_model(model, dataloaders, optimizer, scheduler, args.epochs2, out_dir,
                               args.model_name + '_stage2', logger, 2, args.save_every, args.log_every)

    logger.info('Done. Checkpoints in {}. To use one with detection_inference.py, copy it to '
                '{}/{}.pth or set cfg.model_name.'.format(out_dir, cfg.model_dir, cfg.model_name))


if __name__ == '__main__':
    main()
