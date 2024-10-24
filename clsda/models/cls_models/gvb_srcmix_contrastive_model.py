import torch
import torch.nn.functional as F
from fastda.models import MODELS
from .srcmix_contrastive_model import SrcMixContrastiveModel


@MODELS.register_module(name='gvb_srcmix_contrastive_model')
class GVBSrcMixContrastiveModel(SrcMixContrastiveModel):
    def __init__(self, model_dict, classifier_dict, num_class, low_dim=128,
                 model_moving_average_decay=0.99,
                 proto_moving_average_decay=0.99,
                 fusion_type='reconstruct_double_detach', normalize=True,
                 extra_bank_size=128, high_img_size=1, low_prob_size=4096, high_bank_size=4,
                 no_sorted_beta=False,
                 forward_twice=False, use_orig_src_feature=False, use_tgt_high_low_mix=False,
                 mixup_sample_type='low',
                 ):
        super(GVBSrcMixContrastiveModel, self).__init__(model_dict, classifier_dict, num_class, low_dim=low_dim,
                                                        model_moving_average_decay=model_moving_average_decay,
                                                        proto_moving_average_decay=proto_moving_average_decay,
                                                        fusion_type=fusion_type, normalize=normalize,
                                                        extra_bank_size=extra_bank_size, high_img_size=high_img_size,
                                                        low_prob_size=low_prob_size, high_bank_size=high_bank_size,
                                                        use_orig_src_feature=use_orig_src_feature,
                                                        use_tgt_high_low_mix=use_tgt_high_low_mix,
                                                        no_sorted_beta=no_sorted_beta,
                                                        mixup_sample_type=mixup_sample_type)
        # 强弱图像分别前传
        self.forward_twice = forward_twice

    def test_forward(self, x1_img, **kwargs):
        feat = self.online_network(x1_img, **kwargs)
        online_pred = self.online_classifier(feat)
        target_feat = self.target_network(x1_img, **kwargs)
        target_pred = self.target_classifier(target_feat)
        return online_pred, target_pred

    def model_forward(self, x1_img, x2_img, model_type=None, shuffle_bn=False, **kwargs):
        if model_type == "online":
            feature_extractor = self.online_network
            classifier = self.online_classifier
        elif model_type == 'target':
            feature_extractor = self.target_network
            classifier = self.target_classifier
        else:
            raise RuntimeError('wrong model type specified')
        if not self.forward_twice:
            x1_shape = x1_img.shape[0]
            img_concat = torch.cat((x1_img, x2_img))
            if shuffle_bn:
                img_concat, idx_unshuffle = self._batch_shuffle_ddp(img_concat)
            feat = feature_extractor(img_concat)
            if shuffle_bn:
                feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
            logits, focal_output = classifier(feat)
            prob = F.softmax(logits, dim=-1)
            contrastive_feat = self.feat_prob_fusion(feat, prob, model=model_type)
            contrastive_feat = F.normalize(contrastive_feat, dim=1) if self.normalize else contrastive_feat
            #
            strong_logits = logits[0:x1_shape]
            weak_logits = logits[x1_shape:]
            strong_prob = prob[0:x1_shape]
            weak_prob = prob[x1_shape:]
            strong_contrastive_feat = contrastive_feat[0:x1_shape]
            weak_contrastive_feat = contrastive_feat[x1_shape:]
            strong_focal = focal_output[0:x1_shape]
            weak_focal = focal_output[x1_shape:]
        else:
            # 强增强图像前传
            if shuffle_bn:
                strong_img, idx_unshuffle = self._batch_shuffle_ddp(x1_img)
            else:
                strong_img = x1_img
            feat = feature_extractor(strong_img)
            if shuffle_bn:
                feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
            strong_logits, strong_focal = classifier(feat)
            strong_prob = F.softmax(strong_logits, dim=-1)
            strong_contrastive_feat = self.feat_prob_fusion(feat, strong_prob, model=model_type)
            strong_contrastive_feat = F.normalize(strong_contrastive_feat,
                                                  dim=1) if self.normalize else strong_contrastive_feat
            # 弱增强图像前传
            if shuffle_bn:
                weak_img, idx_unshuffle = self._batch_shuffle_ddp(x2_img)
            else:
                weak_img = x2_img
            feat = feature_extractor(weak_img)
            if shuffle_bn:
                feat = self._batch_unshuffle_ddp(feat, idx_unshuffle)
            weak_logits, weak_focal = classifier(feat)
            weak_prob = F.softmax(weak_logits, dim=-1)
            weak_contrastive_feat = self.feat_prob_fusion(feat, weak_prob, model=model_type)
            weak_contrastive_feat = F.normalize(weak_contrastive_feat,
                                                dim=1) if self.normalize else weak_contrastive_feat
        output = {
            'strong_logits': strong_logits,
            'weak_logits': weak_logits,
            'strong_prob': strong_prob,
            'weak_prob': weak_prob,
            'strong_contrastive_feat': strong_contrastive_feat,
            'weak_contrastive_feat': weak_contrastive_feat,
            'strong_focal': strong_focal,
            'weak_focal': weak_focal,
        }
        return output

    def feat_prob_fusion(self, feat, prob, model='online'):
        # fc_weight = self.online_classifier.fc2.weight.detach()
        classifier = self.online_classifier
        heuristic = classifier.bridge.weight.detach()
        fc_weight = self.online_classifier.fc2.weight.detach() - heuristic
        #
        fc_weight = F.normalize(fc_weight, dim=1) if (not self.normalize) or self.all_normalize else fc_weight
        #
        if self.use_another_temp:
            # print('re representation {}, {}'.format(self.re_representation_temp, self.online_classifier.temp))
            new_prob = F.softmax(feat.mm(fc_weight.detach().t()) / self.re_representation_temp, dim=1)
        else:
            new_prob = F.softmax(feat.mm(fc_weight.detach().t()) / self.online_classifier.temp, dim=1)
        return new_prob.mm(fc_weight)
