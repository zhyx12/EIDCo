import random

import torch
from fastda.models import MODELS
from mmcv.runner import get_dist_info
from fastda.utils import concat_all_gather
from .basic_mix_contrastive_model import BasicMixContrastiveModel
import torch.nn.functional as F


@MODELS.register_module(name='srcmix_contrastive_model')
class SrcMixContrastiveModel(BasicMixContrastiveModel):
    def __init__(self, model_dict, classifier_dict, num_class, low_dim=128,
                 model_moving_average_decay=0.99,
                 proto_moving_average_decay=0.99,
                 fusion_type='reconstruct_double_detach', normalize=True,
                 extra_bank_size=512, high_img_size=1, low_prob_size=4096, high_bank_size=4,
                 no_sorted_beta=False, force_no_shuffle_bn=False, use_tgt_high_low_mix=False,
                 select_topk=None, sample_by_prob=False, use_orig_src_feature=False,
                 mixup_sample_type='low', select_src_by_tgt_similarity=False,
                 src_keep_ratio=0.5,
                 ):
        super(SrcMixContrastiveModel, self).__init__(model_dict, classifier_dict, num_class, low_dim=low_dim,
                                                     model_moving_average_decay=model_moving_average_decay,
                                                     proto_moving_average_decay=proto_moving_average_decay,
                                                     fusion_type=fusion_type, normalize=normalize,
                                                     force_no_shuffle_bn=force_no_shuffle_bn,
                                                     mixup_sample_type=mixup_sample_type,
                                                     select_src_by_tgt_similarity=select_src_by_tgt_similarity,
                                                     src_keep_ratio=src_keep_ratio,
                                                     )
        self.extra_bank_size = extra_bank_size
        self.high_img_size = high_img_size
        self.high_bank_size = high_bank_size
        self.low_prob_size = low_prob_size
        rank, world_size = get_dist_info()
        self.local_rank = rank
        self.world_size = world_size
        self.no_sorted_beta = no_sorted_beta
        self.use_tgt_high_low_mix = use_tgt_high_low_mix
        self.sample_by_prob = sample_by_prob
        self.use_orig_src_feature = use_orig_src_feature
        self.select_topk = select_topk if select_topk is not None else num_class
        self.register_buffer('weak_extra_memory_bank', tensor=torch.zeros(extra_bank_size, low_dim))
        self.register_buffer('weak_extra_bank_ptr', tensor=torch.zeros(1, dtype=torch.long))
        self.register_buffer('weak_extra_entropy', tensor=torch.ones((extra_bank_size,)))
        self.register_buffer('strong_extra_memory_bank', tensor=torch.zeros(extra_bank_size, low_dim))
        self.register_buffer('strong_extra_bank_ptr', tensor=torch.zeros(1, dtype=torch.long))
        self.register_buffer('strong_extra_entropy', tensor=torch.ones((extra_bank_size,)))
        #
        self.register_buffer('class_weak_img', tensor=torch.zeros(self.num_class, high_img_size, 3, 224, 224))
        self.register_buffer('class_strong_img_1', tensor=torch.zeros(self.num_class, high_img_size, 3, 224, 224))
        self.register_buffer('class_strong_img_2', tensor=torch.zeros(self.num_class, high_img_size, 3, 224, 224))
        self.register_buffer('class_weak_feature_bank', tensor=torch.zeros(self.num_class, high_img_size, low_dim))
        self.register_buffer('class_strong_feature_bank', tensor=torch.zeros(self.num_class, high_img_size, low_dim))
        self.register_buffer('class_bank_ptr', tensor=torch.zeros((num_class,), dtype=torch.long))
        self.register_buffer('class_fill_flag', tensor=torch.zeros((num_class,), dtype=torch.long))
        #
        self.register_buffer('weak_high_memory_bank', tensor=torch.zeros(self.num_class, high_bank_size, low_dim))
        self.register_buffer('weak_high_bank_ptr', tensor=torch.zeros((num_class,), dtype=torch.long))
        self.register_buffer('strong_high_memory_bank', tensor=torch.zeros(self.num_class, high_bank_size, low_dim))
        self.register_buffer('strong_high_bank_ptr', tensor=torch.zeros((num_class,), dtype=torch.long))

    @torch.no_grad()
    def update_bank(self, img, key_features, prob, tgt, mask, memory_type):
        if memory_type == 'weak':
            high_memory_bank = self.weak_high_memory_bank
            high_bank_ptr = self.weak_high_bank_ptr
            extra_memory_bank = self.weak_extra_memory_bank
            extra_bank_ptr = self.weak_extra_bank_ptr
            extra_entropy = self.weak_extra_entropy
        elif memory_type == 'strong':
            high_memory_bank = self.strong_high_memory_bank
            high_bank_ptr = self.strong_high_bank_ptr
            extra_memory_bank = self.strong_extra_memory_bank
            extra_bank_ptr = self.strong_extra_bank_ptr
            extra_entropy = self.strong_extra_entropy
        else:
            raise RuntimeError('wrong memory type {}'.format(memory_type))
        #
        all_key_features = concat_all_gather(key_features)
        all_mask = concat_all_gather(mask)
        # 更新高置信度的图像和特征
        # high_mask = all_mask == 1
        # if torch.sum(high_mask) > 0:
        #     all_tgt = concat_all_gather(tgt)
        #     high_features = all_key_features[high_mask]
        #     high_tgt = all_tgt[high_mask]
        #     unique_class = torch.unique(high_tgt)
        #     #
        #     for tmp_class in unique_class:
        #         tmp_class = tmp_class.to(torch.int)
        #         tmp_class_ind = torch.nonzero(high_tgt == tmp_class).squeeze(1)
        #         tmp_class_num = tmp_class_ind.shape[0]
        #         tmp_class_start_point = int(self.class_bank_ptr[tmp_class])
        #         tmp_class_end_point = min(tmp_class_start_point + tmp_class_num, self.high_img_size)
        #         new_feat = high_features[tmp_class_ind[0:(tmp_class_end_point - tmp_class_start_point)], :]
        #         if memory_type == 'weak':
        #             all_img = concat_all_gather(img)
        #             high_img = all_img[high_mask]
        #             new_img = high_img[tmp_class_ind[0:(tmp_class_end_point - tmp_class_start_point)], :]
        #             self.class_weak_img[tmp_class, tmp_class_start_point:tmp_class_end_point, :, :, :] = new_img
        #             self.class_weak_feature_bank[tmp_class, tmp_class_start_point:tmp_class_end_point, :] = new_feat
        #         else:
        #             all_img_1 = concat_all_gather(img[0])
        #             all_img_2 = concat_all_gather(img[1])
        #             high_img_1 = all_img_1[high_mask]
        #             high_img_1 = high_img_1[tmp_class_ind[0:(tmp_class_end_point - tmp_class_start_point)], :]
        #             high_img_2 = all_img_2[high_mask]
        #             high_img_2 = high_img_2[tmp_class_ind[0:(tmp_class_end_point - tmp_class_start_point)], :]
        #             self.class_strong_img_1[tmp_class, tmp_class_start_point:tmp_class_end_point, :, :, :] = high_img_1
        #             self.class_strong_img_2[tmp_class, tmp_class_start_point:tmp_class_end_point, :, :, :] = high_img_2
        #             self.class_strong_feature_bank[tmp_class, tmp_class_start_point:tmp_class_end_point, :] = new_feat
        #         if tmp_class_end_point == self.high_img_size:
        #             self.class_bank_ptr[tmp_class] = 0
        #             self.class_fill_flag[tmp_class] = 1
        #         else:
        #             self.class_bank_ptr[tmp_class] = tmp_class_end_point
        #         # 更新另外的high_memory_bank
        #         tmp_class_start_point = int(high_bank_ptr[tmp_class])
        #         tmp_class_end_point = min(tmp_class_start_point + tmp_class_num, self.high_bank_size)
        #         new_feat = high_features[tmp_class_ind[0:(tmp_class_end_point - tmp_class_start_point)], :]
        #         high_memory_bank[tmp_class, tmp_class_start_point:tmp_class_end_point, :] = new_feat
        #         if tmp_class_end_point == self.high_bank_size:
        #             high_bank_ptr[tmp_class] = 0
        #         else:
        #             high_bank_ptr[tmp_class] = tmp_class_end_point
        # 更新对应类别的bank
        low_mask = all_mask == 0
        extra_features = all_key_features[low_mask, :]
        # extra_ind = torch.nonzero(all_mask == 0).squeeze(1)
        extra_num = extra_features.shape[0]
        if extra_num > 0:
            all_prob = concat_all_gather(prob)
            all_entropy = - torch.sum(all_prob * torch.log(all_prob + 1e-5), dim=1)
            low_entropy = all_entropy[low_mask]
            start_point = int(extra_bank_ptr)
            end_point = min(start_point + extra_num, self.extra_bank_size)
            extra_memory_bank[start_point:end_point, :] = extra_features[0:(end_point - start_point), :]
            extra_entropy[start_point:end_point] = low_entropy[0:(end_point - start_point)]
            if end_point == self.extra_bank_size:
                # 这种写法，在把self.ptr赋给另外一个变量名的时候，会导致新变量更新，而self.ptr不更新，所以采用直接对tensor内元素赋值
                # extra_bank_ptr = torch.zeros((1,), dtype=torch.long, device=extra_bank_ptr.device)
                extra_bank_ptr[0] = 0
            else:
                extra_bank_ptr += extra_num
            # print('extra sample, num {}, new ptr {}'.format(extra_num,self.extra_bank_ptr[0].item()))

    def negative_logits(self, query_features, memory_type, pseudo_label=None):
        if memory_type == 'weak':
            extra_memory_bank = self.weak_extra_memory_bank
        elif memory_type == 'strong':
            extra_memory_bank = self.strong_extra_memory_bank
        else:
            raise RuntimeError('wrong type of memory_type {}'.format(memory_type))
        #
        temp_memory_bank = extra_memory_bank
        logits_neg = query_features.mm(temp_memory_bank.t())
        return logits_neg

    def select_src_samples_for_mixup(self, prob, src_img_list, src_feat_list, src_gt, beta_sampler=None):
        target_sample_num = prob.shape[0]
        src_sample_num = src_img_list[0].shape[0]
        #
        lam = beta_sampler.sample()[0:target_sample_num]
        lam_matrix = torch.cat((lam, 1 - lam), dim=1)
        if self.no_sorted_beta:
            sorted_lam = lam_matrix
        else:
            sorted_lam, _ = torch.sort(lam_matrix, dim=1, descending=True)
        # 测试baseline
        # tmp_ones = torch.ones((target_sample_num, 1), device=self.gpu_device)
        # tmp_zeros = torch.zeros_like(tmp_ones)
        # sorted_lam = torch.cat((tmp_ones, tmp_zeros), dim=1)
        #
        # tmp_p = random.random()
        # if tmp_p >= 0.5 and torch.sum(self.class_fill_flag) == self.num_class:
        #     class_ind = torch.randint(0, self.num_class, (target_sample_num,), device=self.gpu_device)
        #     inter_class_ind = torch.randint(0, self.high_img_size, (target_sample_num,), device=self.gpu_device)
        #     select_high_weak_img = self.class_weak_img[class_ind, inter_class_ind, :, :, :]
        #     select_high_strong_img_1 = self.class_strong_img_1[class_ind, inter_class_ind, :, :, :]
        #     select_high_strong_img_2 = self.class_strong_img_2[class_ind, inter_class_ind, :, :, :]
        #     select_high_weak_feature = self.class_weak_feature_bank[class_ind, inter_class_ind, :]
        #     select_high_strong_feature = self.class_strong_feature_bank[class_ind, inter_class_ind, :]
        #     img_list = [select_high_strong_img_1, select_high_strong_img_2, select_high_weak_img]
        #     feat_list = [select_high_strong_feature, select_high_strong_feature, select_high_weak_feature]
        #     return img_list, feat_list, sorted_lam
        # else:
        random_ind = torch.randint(0, src_sample_num, (target_sample_num,), device=self.gpu_device)
        select_img_list = [src_img_list[0][random_ind], src_img_list[0][random_ind], src_img_list[1][random_ind]]
        if self.use_orig_src_feature:
            select_feat_list = [src_feat_list[0][random_ind], src_feat_list[0][random_ind],
                                src_feat_list[1][random_ind]]
        else:
            fc_weight = self.target_classifier.fc2.weight
            tmp_feat = fc_weight[src_gt[random_ind]]
            select_feat_list = [tmp_feat, tmp_feat, tmp_feat]
        return select_img_list, select_feat_list, sorted_lam
        #
        # if self.use_tgt_high_low_mix:
        #     if torch.sum(self.class_fill_flag) == self.num_class:
        #         #
        #         sorted_prob, sorted_local_low_prob_index = torch.sort(prob, dim=1, descending=True)
        #         if not self.sample_by_prob:
        #             # 随机采类别
        #             class_ind = torch.randint(0, self.select_topk, (target_sample_num,), device=self.gpu_device)
        #         else:
        #             # 根据概率采样
        #             class_ind = torch.multinomial(sorted_prob, num_samples=1).squeeze()
        #         select_class_ind = torch.gather(sorted_local_low_prob_index, dim=1,
        #                                         index=class_ind.unsqueeze(1)).squeeze(1)
        #         inter_class_ind = torch.randint(0, self.high_img_size, (target_sample_num,), device=self.gpu_device)
        #         select_high_weak_img = self.class_weak_img[select_class_ind, inter_class_ind, :, :, :]
        #         select_high_strong_img_1 = self.class_strong_img_1[select_class_ind, inter_class_ind, :, :, :]
        #         select_high_strong_img_2 = self.class_strong_img_2[select_class_ind, inter_class_ind, :, :, :]
        #         select_high_weak_feature = self.class_weak_feature_bank[select_class_ind, inter_class_ind, :]
        #         select_high_strong_feature = self.class_strong_feature_bank[select_class_ind, inter_class_ind, :]
        #         img_list = [select_high_strong_img_1, select_high_strong_img_2, select_high_weak_img]
        #         feat_list = [select_high_strong_feature, select_high_strong_feature, select_high_weak_feature]
        #         return img_list, feat_list, sorted_lam
        #     else:
        #         # print('not fill yet {}'.format(torch.sum(self.class_fill_flag).item()))
        #         tmp_ones = torch.ones((target_sample_num, 1), device=self.gpu_device)
        #         tmp_zeros = torch.zeros_like(tmp_ones)
        #         sorted_lam = torch.cat((tmp_ones, tmp_zeros), dim=1)
        # # #
        # random_ind = torch.randint(0, src_sample_num, (target_sample_num,), device=self.gpu_device)
        # select_img_list = [src_img_list[0][random_ind], src_img_list[0][random_ind], src_img_list[1][random_ind]]
        # fc_weight = self.target_classifier.fc2.weight
        # tmp_feat = fc_weight[src_gt[random_ind]]
        # select_feat_list = [tmp_feat, tmp_feat, tmp_feat]
        # return select_img_list, select_feat_list, sorted_lam
