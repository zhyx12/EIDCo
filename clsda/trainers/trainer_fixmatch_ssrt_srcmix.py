# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import torch.nn.functional as F
from fastda.runner import BaseTrainer, BaseValidator, TRAINER, VALIDATOR
from fastda.hooks import MetricsLogger
from ..hooks import ClsAccuracy, ClsBestAccuracyByVal
from fastda.utils import get_root_logger
from mmcv.runner import get_dist_info
from fastda.utils import concat_all_gather
from fastada_cls.models.gvb_network import calc_coeff
from fastada_cls.models.hda_network import HDALoss


@VALIDATOR.register_module(name='fixmatchssrtsrcmix')
class ValidatorFixMatchSSRTSrcMix(BaseValidator):
    def __init__(self, basic_parameters):
        super(ValidatorFixMatchSSRTSrcMix, self).__init__(**basic_parameters)

    def eval_iter(self, val_batch_data):
        # val_img, val_label, val_name = val_batch_data
        val_img = val_batch_data['img']
        val_label = val_batch_data['gt_label'].squeeze(1)
        val_metas = val_batch_data['img_metas']
        with torch.no_grad():
            pred_unlabeled, target_logits = self.model_dict['base_model'](val_img)
        return {'img': val_img,
                'gt': val_label,
                'img_metas': val_metas,
                'pred': pred_unlabeled,
                'target_pred': target_logits,
                }


@TRAINER.register_module('fixmatchssrtsrcmix')
class TrainerFixMatchSSRTSrcMix(BaseTrainer):
    def __init__(self, basic_parameters,
                 # new parameters
                 lambda_temp=1.0, lambda_info_nce=1.0, lambda_kld=0.1,
                 prob_threshold=0.8,
                 beta_param=3.0,
                 lambda_fixmatch=1.0,
                 #
                 lambda_adv=1.0,
                 src_mixup=False, src_ce_type='weak', loss_type=0,
                 ):
        super(TrainerFixMatchSSRTSrcMix, self).__init__(**basic_parameters)
        self.lambda_temp = lambda_temp
        self.lambda_info_nce = lambda_info_nce
        self.prob_threshold = prob_threshold
        self.lambda_fixmatch = lambda_fixmatch
        self.lambda_kld = lambda_kld
        self.lambda_adv = lambda_adv
        self.src_mixup = src_mixup
        self.src_ce_type = src_ce_type
        self.high_ratio = None
        self.loss_type = loss_type
        self.num_class = self.train_loaders[0].dataset.n_classes
        #
        rank, world_size = get_dist_info()
        unlabeled_batchsize_all = 36
        labeled_batchsize_all = 36
        self.unlabeled_batchsize = int((unlabeled_batchsize_all) / world_size)
        tmp_tgt_batchsize = int((unlabeled_batchsize_all + labeled_batchsize_all)) * 2
        beta_param = torch.tensor([beta_param, ] * tmp_tgt_batchsize,
                                  device='cuda:{}'.format(self.local_rank)).unsqueeze(1)
        self.beta_sampler = torch.distributions.beta.Beta(beta_param, beta_param)
        self.world_size = world_size
        self.mixup_feature = None
        self.lambda_mixup = None
        # 增加记录
        if self.local_rank == 0:
            log_names = ['cls', 'acc1', 'info_nce', 'pos_max_prob', 'mean_max_prob', 'mask', 'consistency',
                         'lam_top1',
                         ]
            loss_metrics = MetricsLogger(log_names=log_names, group_name='loss', log_interval=self.log_interval)
            self.register_hook(loss_metrics)
        #
        self.main_hist = None

    def train_iter(self, *args):
        if len(args) == 2:
            src_img_weak = args[0][0]['img']
            src_label_weak = args[0][0]['gt_label'].squeeze(1)
            src_img_strong = args[0][1]['img']
            src_label_strong = args[0][1]['gt_label'].squeeze(1)
            # TODO：使用没有tgt labeled的数据集
            tgt_unlabeled_img_weak = args[1][0]['img'].squeeze(0)
            tgt_unlabeled_img_strong = args[1][1]['img'].squeeze(0)
            tgt_unlabeled_img_strong_2 = args[1][2]['img'].squeeze(0)
        elif len(args) == 3:
            src_img_weak = args[0][0]['img']
            src_label_weak = args[0][0]['gt_label'].squeeze(1)
            src_img_strong = args[0][1]['img']
            src_label_strong = args[0][1]['gt_label'].squeeze(1)
            #
            tgt_labeled_img_weak = args[1][0]['img']
            tgt_labeled_gt_weak = args[1][0]['gt_label'].squeeze(1)
            tgt_label_img_strong = args[1][1]['img']
            #
            src_img_weak = torch.cat((src_img_weak, tgt_labeled_img_weak), dim=0)
            src_img_strong = torch.cat((src_img_strong, tgt_label_img_strong), dim=0)
            src_label_strong = torch.cat((src_label_strong, tgt_labeled_gt_weak), dim=0)
            # TODO：使用没有tgt labeled的数据集
            tgt_unlabeled_img_weak = args[2][0]['img'].squeeze(0)
            tgt_unlabeled_img_strong = args[2][1]['img'].squeeze(0)
            tgt_unlabeled_img_strong_2 = args[2][2]['img'].squeeze(0)
        else:
            raise RuntimeError('wrong len of data')
        #
        src_labeled_size = src_img_strong.shape[0]
        tgt_unlabeled_size = tgt_unlabeled_img_weak.shape[0]
        contrastive_size = tgt_unlabeled_size
        #
        batch_metrics = {}
        batch_metrics['loss'] = {}
        #
        base_model = self.model_dict['base_model']
        #
        self.zero_grad_all()
        #
        all_labeled_img = torch.cat((src_img_strong, tgt_unlabeled_img_strong, tgt_unlabeled_img_strong_2), 0)
        all_tgt_img = torch.cat((src_img_weak, tgt_unlabeled_img_weak), 0)
        #
        tmp_res = base_model(all_labeled_img, all_tgt_img, src_mixup=self.src_mixup, threshold=self.prob_threshold,
                             src_labeled_size=src_labeled_size,
                             tgt_unlabeled_size=tgt_unlabeled_size, beta_sampler=self.beta_sampler,
                             src_gt=src_label_strong,src_labels=src_label_strong,)
        online_output, target_output, lambda_mixup, mixup_feature = tmp_res
        self.lambda_mixup = lambda_mixup
        self.mixup_feature = mixup_feature
        online_strong_logits = online_output['strong_logits']
        target_weak_logits = target_output['weak_logits']
        online_weak_logits = online_output['weak_logits']
        online_strong_contrastive_feat = online_output['strong_contrastive_feat']
        online_weak_contrastive_feat = online_output['weak_contrastive_feat']
        target_strong_contrastive_feat = target_output['strong_contrastive_feat']
        target_weak_contrastive_feat = target_output['weak_contrastive_feat']
        dis_output = online_output['dis_output']
        #
        domain_labels = torch.cat((torch.zeros(src_labeled_size, device=dis_output.device, dtype=torch.float),
                                   torch.ones(tgt_unlabeled_size, device=dis_output.device, dtype=torch.float)),
                                  dim=0)
        loss_adv = F.binary_cross_entropy(
            torch.sigmoid(dis_output.squeeze()[0:(src_labeled_size + tgt_unlabeled_size)]),
            domain_labels)
        # 监督损失
        if self.src_ce_type == 'weak':
            loss_supervised = F.cross_entropy(online_weak_logits[0:src_labeled_size], src_label_strong)
        elif self.src_ce_type == 'strong':
            loss_supervised = F.cross_entropy(online_strong_logits[0:src_labeled_size], src_label_strong)
        else:
            raise RuntimeError('wrong type of {}'.format("src_ce_type"))
        # loss_supervised = F.cross_entropy(online_weak_logits[0:src_labeled_size], src_label_strong)
        loss = loss_supervised + self.lambda_adv * loss_adv * 2.0
        # #
        # # # 输出的一致性损失
        pseudo_label = torch.softmax(
            target_weak_logits[src_labeled_size:(src_labeled_size + tgt_unlabeled_size)].detach(),
            dim=-1)
        strong_aug_pred = online_strong_logits[src_labeled_size:(src_labeled_size + tgt_unlabeled_size)]
        #
        max_probs, tgt_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(self.prob_threshold).float().detach()
        #
        loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
        loss += loss_consistency * self.lambda_fixmatch
        loss_consistency_val = loss_consistency.item()
        # kld reg
        logsoftmax = F.log_softmax(strong_aug_pred, dim=1)
        kld = torch.sum(-logsoftmax / self.num_class, dim=1)
        loss += (self.lambda_kld * kld * mask).mean()
        # #
        mask_val = torch.sum(mask).item() / mask.shape[0]
        self.high_ratio = mask_val if self.high_ratio is None else mask_val
        # # ############################################################################################
        # # # 更新memory只用新的数据
        feat_for_update = target_weak_contrastive_feat[src_labeled_size:(src_labeled_size + tgt_unlabeled_size), :]
        weak_img_for_udpate = tgt_unlabeled_img_weak
        prob_for_update = pseudo_label
        strong_img_for_update = (tgt_unlabeled_img_strong, tgt_unlabeled_img_strong_2)
        strong_feat_for_update = target_strong_contrastive_feat[
                                 src_labeled_size:(src_labeled_size + tgt_unlabeled_size), :]
        mask_for_update = mask[0:tgt_unlabeled_size]
        tgt_for_update = tgt_u[0:tgt_unlabeled_size]
        # #
        # # constrastive loss
        all_k_strong = target_strong_contrastive_feat
        all_k_weak = target_weak_contrastive_feat
        weak_feat_for_backbone = online_weak_contrastive_feat[src_labeled_size:]
        k_weak_for_backbone = all_k_weak[src_labeled_size:]
        k_strong_for_backbone = all_k_strong[src_labeled_size:(src_labeled_size + contrastive_size)]
        strong_feat_for_backbone = online_strong_contrastive_feat[
                                   src_labeled_size:(src_labeled_size + contrastive_size)]
        k_strong_2 = all_k_strong[(src_labeled_size + contrastive_size):]
        feat_strong_2 = online_strong_contrastive_feat[(src_labeled_size + contrastive_size):]
        # ###
        # symmetric loss
        if self.loss_type == 0:
            info_nce_loss_1 = self.contrastive_loss(strong_feat_for_backbone, k_strong_2,
                                                    tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=0)
            info_nce_loss_2 = self.contrastive_loss(feat_strong_2, k_strong_for_backbone,
                                                    tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=1)
            info_nce_loss = (info_nce_loss_1 + info_nce_loss_2) / 2.0
        elif self.loss_type == 1:
            info_nce_loss_1 = self.contrastive_loss(strong_feat_for_backbone, k_weak_for_backbone, tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=2)
            info_nce_loss_3 = self.contrastive_loss(strong_feat_for_backbone, k_strong_2, tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=1)
            info_nce_loss_2 = self.contrastive_loss(weak_feat_for_backbone, k_strong_for_backbone, tgt_u,
                                                    mask, memory_type='weak', ind=0)
            info_nce_loss = (info_nce_loss_1 + info_nce_loss_2 + info_nce_loss_3) / 3.0
        elif self.loss_type == 2:
            info_nce_loss_1 = self.contrastive_loss(strong_feat_for_backbone, k_strong_2,
                                                    tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=0)
            info_nce_loss_2 = self.contrastive_loss(feat_strong_2, k_strong_for_backbone,
                                                    tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=1)
            info_nce_loss_3 = self.contrastive_loss(strong_feat_for_backbone, k_weak_for_backbone, tgt_u,
                                                    mask, batch_metrics, memory_type='strong', ind=2)
            info_nce_loss_4 = self.contrastive_loss(weak_feat_for_backbone, k_strong_for_backbone, tgt_u,
                                                    mask, memory_type='weak', ind=0)
            info_nce_loss = (info_nce_loss_1 + info_nce_loss_2 + info_nce_loss_3 + info_nce_loss_4) / 4.0
        else:
            raise ValueError('wrong loss type')
        #
        loss += info_nce_loss * self.lambda_info_nce
        #
        loss.backward()
        base_model.module.update_bank(weak_img_for_udpate, feat_for_update, prob_for_update, tgt_for_update,
                                      mask_for_update, memory_type='weak')
        base_model.module.update_bank(strong_img_for_update, strong_feat_for_update, prob_for_update, tgt_for_update,
                                      mask_for_update, memory_type='strong')
        #
        self.step_grad_all()
        #
        batch_metrics['loss']['cls'] = loss_supervised.item()
        batch_metrics['loss']['consistency'] = loss_consistency_val
        batch_metrics['loss']['info_nce'] = info_nce_loss.item() if isinstance(info_nce_loss,
                                                                               torch.Tensor) else info_nce_loss
        batch_metrics['loss']['mean_max_prob'] = torch.mean(max_probs).item()
        batch_metrics['loss']['mask'] = mask_val
        if self.lambda_mixup is not None:
            sorted_lam, _ = torch.sort(self.lambda_mixup, dim=1, descending=True)
            cumsum_lam = torch.cumsum(sorted_lam, dim=1)
            batch_metrics['loss']['lam_top1'] = torch.mean(cumsum_lam[:, 0]).item()
        else:
            batch_metrics['loss']['lam_top1'] = 0
        with torch.no_grad():
            pos_prob = F.softmax(online_strong_logits[src_labeled_size:(src_labeled_size + tgt_unlabeled_size)], dim=1)
            pos_mean_max_prob = torch.max(pos_prob, dim=1)[0].mean()

        batch_metrics['loss']['pos_max_prob'] = pos_mean_max_prob.item()
        return batch_metrics

    def load_pretrained_model(self, weights_path):
        logger = get_root_logger()
        weights = torch.load(weights_path, map_location='cpu')
        weights = weights['base_model']
        self.model_dict['base_model'].load_state_dict(weights)
        logger.info('load pretrained model {}'.format(weights_path))

    def contrastive_loss(self, query_feat, key_feat, pseudo_label, mask, batch_metrics=None, memory_type='weak',
                         ind=None):
        low_conf_ind = mask == 0
        low_conf_query_feat = query_feat[low_conf_ind, :]
        low_conf_key_feat = key_feat[low_conf_ind, :]
        low_conf_num = low_conf_query_feat.shape[0]
        #
        low_logits, low_conf_nce_loss = self.single_contrastive_loss(low_conf_query_feat, low_conf_key_feat,
                                                                     pseudo_label,
                                                                     memory_type=memory_type, mixup_feat_ind=ind)
        #
        nce_loss = low_conf_nce_loss * (1 - self.high_ratio)
        #
        if batch_metrics is not None:
            # 计算instance对比学习中的准确率
            if low_conf_num > 0:
                final_logits = low_logits
                _, pred_max_ind = torch.max(final_logits, dim=1)
                #
                final_gt = self.get_contrastive_labels(pred_max_ind)
                acc1 = torch.mean((pred_max_ind == final_gt).float())
                batch_metrics['loss']['acc1'] = acc1.item()
            else:
                print('meet no high and no low situtation')
                batch_metrics['loss']['acc1'] = 0
                return nce_loss
        #
        return nce_loss

    def single_contrastive_loss(self, query_feat, key_feat, pseudo_label, memory_type, mixup_feat_ind=None):
        sample_num = query_feat.shape[0]
        if sample_num == 0:
            return None, 0
        if self.src_mixup:
            tmp_key_feat = self.mixup_feature[mixup_feat_ind]
            l_pos_high = torch.sum(query_feat.unsqueeze(1) * tmp_key_feat, dim=2)
        else:
            tmp_key_feat = key_feat
            l_pos_high = torch.einsum('nc,nc->n', [query_feat, tmp_key_feat]).unsqueeze(1)
        #
        l_neg_high = self.model_dict['base_model'].module.negative_logits(query_feat, memory_type=memory_type,
                                                                          pseudo_label=pseudo_label)
        logits_1 = torch.cat((l_pos_high, l_neg_high), dim=1) / self.lambda_temp
        #
        constrastive_labels = self.get_contrastive_labels(query_feat)
        if self.src_mixup:
            lam = self.lambda_mixup
            info_nce_loss = lam[:, 0] * F.cross_entropy(logits_1, constrastive_labels, reduction='none')
            info_nce_loss += lam[:, 1] * F.cross_entropy(logits_1, constrastive_labels + 1, reduction='none')
            info_nce_loss = torch.mean(info_nce_loss)
        else:
            info_nce_loss = F.cross_entropy(logits_1, constrastive_labels)
        return logits_1, info_nce_loss

    def get_contrastive_labels(self, query_feat):
        current_batch_size = query_feat.shape[0]
        constrastive_labels = torch.zeros((current_batch_size,), dtype=torch.long,
                                          device='cuda:{}'.format(self.local_rank))
        return constrastive_labels
