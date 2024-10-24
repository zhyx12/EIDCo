
import torch
import torch.nn.functional as F
from mmcv.runner.hooks import Hook
from clsda.utils.metrics import runningMetric
from clsda.utils import get_root_logger, get_root_writer
from mmcv.runner import get_dist_info
import pickle
import numpy as np
import os
import math
from clsda.utils.utils import concat_all_gather


class ClsUniformityTolerance(Hook):
    def __init__(self, runner, dataset_name, pred_key='pred', feat_dim=512, threshold=0.8):
        rank, world_size = get_dist_info()
        self.local_rank = rank
        self.world_size = world_size
        self.dataset_name = dataset_name
        self.pred_key = pred_key
        self.feat_dim = feat_dim
        self.threshold = threshold
        #

    def before_val_epoch(self, runner):
        dataset_len = len(runner.test_loaders[self.dataset_name].dataset)
        # print('len is {}'.format(dataset_len))
        gpu_device = "cuda:{}".format(self.local_rank)
        self.reconstruct_feature = torch.zeros((int(dataset_len / self.world_size), self.feat_dim), device=gpu_device)
        self.orig_feature = torch.zeros((int(dataset_len / self.world_size), self.feat_dim), device=gpu_device)
        self.gt_label = torch.zeros((int(dataset_len / self.world_size),), device=gpu_device)
        self.low_indicator = torch.zeros((int(dataset_len / self.world_size),), dtype=torch.bool, device=gpu_device)
        # print('feat shape {}'.format(self.feature.shape))
        # exit(0)
        self.pointer = 0

    def after_val_iter(self, runner):
        batch_output = runner.batch_output
        dataset_name = batch_output['dataset_name']
        if dataset_name == self.dataset_name:
            gt = batch_output['gt']
            orig_feature = batch_output['orig_feature']
            reconstruct_feature = batch_output['reconstruct_feature']
            pred = batch_output[self.pred_key]
            prob = F.softmax(pred, dim=1)
            max_prob, pred_ind = torch.max(prob, dim=1)
            #
            all_shape = self.orig_feature.shape[0]
            if self.pointer <= all_shape and self.pointer + orig_feature.shape[0] <= all_shape:
                new_pointer = self.pointer + orig_feature.shape[0]
                self.orig_feature[self.pointer:new_pointer, :] = orig_feature
                self.reconstruct_feature[self.pointer:new_pointer, :] = reconstruct_feature
                self.gt_label[self.pointer:new_pointer] = gt
                self.low_indicator[self.pointer:new_pointer] = max_prob < self.threshold
                self.pointer = new_pointer
                print('pointer {}'.format(self.pointer))

    def after_val_epoch(self, runner):
        #
        logger = get_root_logger()
        writer = get_root_writer()
        #
        tmp_low_indicator = self.low_indicator[0:self.pointer]
        orig_feature = self.orig_feature[0:self.pointer]
        reconstruct_feature = self.reconstruct_feature[0:self.pointer]
        all_label = self.gt_label[0:self.pointer]
        # concat all gather
        orig_feature = concat_all_gather(orig_feature)
        reconstruct_feature = concat_all_gather(reconstruct_feature)
        all_label = concat_all_gather(all_label)
        tmp_low_indicator = concat_all_gather(tmp_low_indicator)
        print('low ratio {}'.format(torch.mean(tmp_low_indicator.to(torch.float))))
        tmp_high_indicator = (torch.ones_like(tmp_low_indicator).float() - tmp_low_indicator.float()).bool()
        #
        if self.local_rank == 0:
            low_label = all_label[tmp_low_indicator]
            high_label = all_label[tmp_high_indicator]
            orig_low_feature = orig_feature[tmp_low_indicator]
            orig_high_feature = orig_feature[tmp_high_indicator]
            reconstruct_low_feature = reconstruct_feature[tmp_low_indicator]
            #
            orig_uniformity_all = uniformity_cal_all(orig_feature)
            orig_tolerance_all = tolerance_cal_all(orig_feature, all_label)
            orig_reverse_tolerance_all = reverse_tolerance_cal_all(orig_feature, all_label)
            reconstruct_uniformity_all = uniformity_cal_all(reconstruct_feature)
            reconstruct_tolerance_all = tolerance_cal_all(reconstruct_feature, all_label)
            reconstruct_reverse_tolerance_all = reverse_tolerance_cal_all(reconstruct_feature, all_label)
            #
            orig_uniformity_low = uniformity_cal_all(orig_low_feature)
            orig_tolerance_low = tolerance_cal_all(orig_low_feature, low_label)
            orig_reverse_tolerance_low = reverse_tolerance_cal_all(orig_low_feature, low_label)
            reconstruct_uniformity_low = uniformity_cal_all(reconstruct_low_feature)
            reconstruct_tolerance_low = tolerance_cal_all(reconstruct_low_feature, low_label)
            reconstruct_reverse_tolerance_low = reverse_tolerance_cal_all(reconstruct_low_feature, low_label)
            #
            orig_tolerance_high = tolerance_cal_all(orig_high_feature, high_label)
            orig_reverse_tolerance_high = reverse_tolerance_cal_all(orig_high_feature, high_label)
            print('all {}, {}'.format(orig_tolerance_all,orig_reverse_tolerance_all))
            print('low {}, {}'.format(orig_tolerance_low,orig_reverse_tolerance_low ))
            print('high {}, {}'.format(orig_tolerance_high, orig_reverse_tolerance_high))
            #
            writer.add_scalar('contrastive_analysis/all_orig_uniformity', orig_uniformity_all.item() * 100,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/all_orig_tolerance', orig_tolerance_all.item() * 1000,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/all_orig_reverse_tolerance',
                              orig_reverse_tolerance_all.item() * 1000,
                              global_step=runner.iteration)
            #
            writer.add_scalar('contrastive_analysis/all_reconstruct_uniformity',
                              reconstruct_uniformity_all.item() * 100,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/all_reconstruct_tolerance', reconstruct_tolerance_all.item() * 1000,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/all_reconstruct_reverse_tolerance',
                              reconstruct_reverse_tolerance_all.item() * 1000,
                              global_step=runner.iteration)
            #
            writer.add_scalar('contrastive_analysis/low_orig_uniformity', orig_uniformity_low.item() * 100,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/low_orig_tolerance', orig_tolerance_low.item() * 1000,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/low_orig_reverse_tolerance',
                              orig_reverse_tolerance_low.item() * 1000,
                              global_step=runner.iteration)
            #
            writer.add_scalar('contrastive_analysis/low_reconstruct_uniformity',
                              reconstruct_uniformity_low.item() * 100,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/low_reconstruct_tolerance', reconstruct_tolerance_low.item() * 1000,
                              global_step=runner.iteration)
            writer.add_scalar('contrastive_analysis/low_reconstruct_reverse_tolerance',
                              reconstruct_reverse_tolerance_low.item() * 1000,
                              global_step=runner.iteration)
        # feat_save_path = os.path.join(runner.logdir,
        #                               'iter_{}_feat_{}_part_{}.pkl'.format(runner.iteration, self.feat_name,
        #                                                                    self.local_rank))
        # with open(feat_save_path, 'wb') as f:
        #     pickle.dump(self.feature[0:self.pointer, :][tmp_low_indicator], f)
        # gt_save_path = os.path.join(runner.logdir,
        #                             'iter_{}_gt_{}_part_{}.pkl'.format(runner.iteration, self.feat_name,
        #                                                                self.local_rank))
        # with open(gt_save_path, 'wb') as f:
        #     pickle.dump(self.gt_label[0:self.pointer][tmp_low_indicator], f)
        #
        self.feature = None
        self.pointer = None
        self.gt_label = None
        #


def uniformity_cal_all(feature, t=2, chunk_size=800):
    num_feat = feature.shape[0]
    part_num = math.ceil(num_feat / chunk_size)
    all_val = torch.zeros((part_num, part_num))
    for i in range(part_num):
        for j in range(part_num):
            all_val[i, j] = uniformity_cal_part(feature[i * chunk_size:(i + 1) * chunk_size],
                                                feature[j * chunk_size:(j + 1) * chunk_size], t=t)
    uniformity = torch.mean(all_val)
    return uniformity


def uniformity_cal_part(feature_1, feature_2, t=2):
    normalized_feature_1 = F.normalize(feature_1, dim=1, p=2)
    normalized_feature_2 = F.normalize(feature_2, dim=1, p=2)
    feature_1 = normalized_feature_1.unsqueeze(dim=1)
    feature_2 = normalized_feature_2.unsqueeze(dim=0)
    uniformity = torch.exp(-torch.linalg.norm(feature_1 - feature_2, dim=2) * t)
    return torch.mean(uniformity)


def tolerance_cal_all(feature, label, chunk_size=800):
    class_num = int((torch.max(label) + 1).item())
    class_ratio = torch.zeros((class_num,))
    class_tolerance = torch.zeros((class_num,))
    for k in range(class_num):
        class_mask = label == k
        class_feat = feature[class_mask]
        class_label = label[class_mask]
        class_tolerance[k] = reverse_tolerance_cal_all(class_feat, class_label, chunk_size=chunk_size, reverse=False)
        class_ratio[k] = torch.pow(torch.sum(class_mask), 2)
    class_ratio = F.normalize(class_ratio, p=1, dim=0)
    tolerance = torch.mean(class_ratio.to(class_tolerance.device) * class_tolerance)
    return tolerance


def reverse_tolerance_cal_all(feature, label, chunk_size=800, reverse=True):
    num_feat = feature.shape[0]
    part_num = math.ceil(num_feat / chunk_size)
    all_val = torch.zeros((part_num, part_num))
    for i in range(part_num):
        for j in range(part_num):
            feat_1 = feature[i * chunk_size:(i + 1) * chunk_size, :]
            label_1 = label[i * chunk_size:(i + 1) * chunk_size]
            feat_2 = feature[j * chunk_size:(j + 1) * chunk_size, :]
            label_2 = label[j * chunk_size:(j + 1) * chunk_size]
            all_val[i, j] = reverse_tolerance_cal_part(feat_1, feat_2, label_1, label_2, reverse=reverse)
    reverse_tolerance = torch.mean(all_val)
    # print("reverse_tolerance is {}".format(reverse_tolerance))
    return reverse_tolerance


def reverse_tolerance_cal_part(feature_1, feature_2, label_1, label_2, reverse=True):
    num_feat = feature_1.shape[0]
    normalized_feature_1 = F.normalize(feature_1, dim=1, p=2)
    normalized_feature_2 = F.normalize(feature_2, dim=1, p=2)
    feature_1 = normalized_feature_1.unsqueeze(dim=1)
    feature_2 = normalized_feature_2.unsqueeze(dim=0)
    feature_cos_similarity = torch.sum(feature_1 * feature_2, dim=2)
    label_1 = label_1.unsqueeze(1)
    label_2 = label_2.unsqueeze(0)
    if reverse:
        same_class_mask = label_1 != label_2
    else:
        same_class_mask = label_1 == label_2
    select_cos_similarity = feature_cos_similarity[same_class_mask]
    tolerance = torch.sum(select_cos_similarity) / torch.sum(same_class_mask).float()
    # print('all shape {}, mask num {}'.format(num_feat*num_feat, torch.sum(same_class_mask) ))
    return tolerance
