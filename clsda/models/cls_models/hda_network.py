# Author: Vincent Zhang
# Mail: zhyx12@gmail.com
# ----------------------------------------------
import torch
import torch.nn as nn
from fastda.models import MODELS
import torch.nn.functional as F
from .basenet import grad_reverse
from .gvb_network import Entropy, Myloss
import numpy as np
from torchvision import models


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        print('init linear layer!!!!!!!!!!!!!!!!!')
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def hun_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        print('init linear layer by hun weight !!!!!!!!!!!!!!!!!')
        nn.init.kaiming_normal_(m.weight, a=100)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def one_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        print('init linear layer by one weight !!!!!!!!!!!!!!!!!')
        nn.init.kaiming_normal_(m.weight, a=1)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def two_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        print('init linear layer by two weight !!!!!!!!!!!!!!!!!')
        nn.init.kaiming_normal_(m.weight, a=2)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def three_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=3)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def four_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=4)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


@MODELS.register_module()
class HDAClassifier(nn.Module):
    def __init__(self, num_class=64, inc=512, temp=0.04):
        super(HDAClassifier, self).__init__()
        self.fc2 = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        self.fc2.apply(hun_weights)
        #
        self.heuristic = nn.Linear(inc, num_class, bias=False)
        self.heuristic.apply(init_weights)
        self.heuristic1 = nn.Linear(inc, num_class, bias=False)
        self.heuristic1.apply(one_weights)
        self.heuristic2 = nn.Linear(inc, num_class, bias=False)
        self.heuristic2.apply(two_weights)

    def forward(self, x, gvbg=True, normalize=False, toalign=False, labels=None, reverse=False, eta=1.0):
        if normalize:
            x = F.normalize(x)
        if not toalign:
            x_out = self.fc2(x) / self.temp
            #
            now1 = self.heuristic(x) / self.temp
            now2 = self.heuristic1(x) / self.temp
            now3 = self.heuristic2(x) / self.temp
            # now_all = torch.cat((now1, now2, now3), 0).reshape(self.heuristic_num, -1, now1.shape[1])
            geuristic = (now1 + now2 + now3)
            #
            if gvbg:
                x_out = x_out - geuristic
            return x_out, geuristic
        else:
            w_pos = self._get_toalign_weight(x, labels)
            x_pos = x * w_pos
            x_out = self.fc2(x_pos) / self.temp
            #
            now1 = self.heuristic(x_pos) / self.temp
            now2 = self.heuristic1(x_pos) / self.temp
            now3 = self.heuristic2(x_pos) / self.temp
            # now_all = torch.cat((now1, now2, now3), 0).reshape(self.heuristic_num, -1, now1.shape[1])
            geuristic = (now1 + now2 + now3)
            #
            if gvbg:
                x_out = x_out - geuristic
            return x_out, geuristic

    def _get_toalign_weight(self, f, labels=None):
        assert labels is not None, f'labels should be asigned'
        w = self.fc2.weight[labels].detach()  # [B, C]
        w0 = self.heuristic.weight[labels].detach()
        w1 = self.heuristic1.weight[labels].detach()
        w2 = self.heuristic2.weight[labels].detach()
        w = w - (w0 + w1 + w2)
        eng_org = (f ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        eng_aft = ((f * w) ** 2).sum(dim=1, keepdim=True)  # [B, 1]
        scalar = (eng_org / eng_aft).sqrt()
        w_pos = w * scalar

        return w_pos

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                optim_param.append({'params': param, 'lr': lr})
                print('GVB Classifier {} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))
        return optim_param


def HDALoss(input_list, ad_net, coeff=None, myloss=Myloss(), iteration=None, split_num=None, domain_label=None,
            target_domain_label=1):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out, _ = ad_net(softmax_output, iteration=iteration)
    batch_size = softmax_output.size(0)
    if split_num == None:
        split_num = batch_size // 2
    #
    x = softmax_output
    entropy = Entropy(x)
    entropy = grad_reverse(entropy, lambd=coeff)
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    heuristic = torch.mean(torch.abs(focals))

    source_mask = torch.ones_like(entropy)
    source_mask[split_num:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:split_num] = 0
    target_weight = entropy * target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    if domain_label is not None:
        # dc_target = torch.from_numpy(np.array([0] * split_num + [1] * (batch_size - split_num))).long().to(
        # softmax_output.device)
        target_label = torch.from_numpy(np.array([target_domain_label] * (batch_size - split_num))).long().to(
            softmax_output.device)
        dc_target = torch.cat((domain_label, target_label))
        adv_loss = torch.sum(F.cross_entropy(ad_out, dc_target, reduction='none') * weight)
        return adv_loss, mean_entropy, heuristic
    else:
        ad_out = nn.Sigmoid()(ad_out)
        dc_target = torch.from_numpy(np.array([[0]] * split_num + [[1]] * (batch_size - split_num))).float().to(
            softmax_output.device)
        return myloss(ad_out, dc_target, weight.view(-1, 1)), mean_entropy, heuristic


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


@MODELS.register_module()
class HDAResNetFc(nn.Module):
    def __init__(self, resnet_name, bottleneck_dim=256, new_cls=True, class_num=1000, heuristic_num=3,
                 heuristic_initial=True,
                 temp=0.03):
        super(HDAResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.temp = temp
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        # self.select_layers = nn.Sequential(self.layer3, self.layer4, self.avgpool)

        self.sigmoid = nn.Sigmoid()
        self.new_cls = new_cls
        self.heuristic_num = heuristic_num
        if new_cls:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            if heuristic_initial:
                self.fc.apply(hun_weights)
            else:
                self.fc.apply(init_weights)
            self.heuristic = nn.Linear(model_resnet.fc.in_features, class_num)
            self.heuristic.apply(init_weights)
            self.heuristic1 = nn.Linear(model_resnet.fc.in_features, class_num)
            self.heuristic1.apply(one_weights)
            self.heuristic2 = nn.Linear(model_resnet.fc.in_features, class_num)
            self.heuristic2.apply(two_weights)
            # self.heuristic3 = nn.Linear(model_resnet.fc.in_features, class_num)
            # self.heuristic3.apply(three_weights)
            # self.heuristic4 = nn.Linear(model_resnet.fc.in_features, class_num)
            # self.heuristic4.apply(four_weights)
            self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x, heuristic=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        #
        if self.heuristic_num == 1:
            geuristic = self.heuristic(x)
        elif self.heuristic_num == 2:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now_all = torch.cat((now1, now2), 0).reshape(self.heuristic_num, -1, now1.shape[1])
            geuristic = now1 + now2
        elif self.heuristic_num == 3:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now_all = torch.cat((now1, now2, now3), 0).reshape(self.heuristic_num, -1, now1.shape[1])
            geuristic = (now1 + now2 + now3)
        elif self.heuristic_num == 4:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now4 = self.heuristic3(x)
            geuristic = (now1 + now2 + now3 + now4)
        elif self.heuristic_num == 5:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now4 = self.heuristic3(x)
            now5 = self.heuristic4(x)
            geuristic = (now1 + now2 + now3 + now4 + now5)
        y = self.fc(x)
        if heuristic:
            y = y - geuristic
        return x, y, geuristic

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'feature_layers' in name or 'conv' in name or 'bn' in name or 'layer' in name:
                    optim_param.append({'params': param, 'lr': lr})
                # print('GVB Classifier {} will be optimized, lr {}'.format(name, lr))
                elif 'fc' in name or 'heuristic' in name:
                    optim_param.append({'params': param, 'lr': lr * 10})
                else:
                    raise RuntimeError('wrong layer name {}'.format(name))
            else:
                print('{} will be ignored'.format(name))
        return optim_param


#
@MODELS.register_module()
class HDAResNetCosine(nn.Module):
    def __init__(self, resnet_name, bottleneck_dim=256, new_cls=True, class_num=1000, heuristic_num=3,
                 heuristic_initial=True,
                 temp=0.04):
        super(HDAResNetCosine, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.temp = temp
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        # self.select_layers = nn.Sequential(self.layer3, self.layer4, self.avgpool)

        self.sigmoid = nn.Sigmoid()
        self.new_cls = new_cls
        self.heuristic_num = heuristic_num
        if new_cls:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num, bias=False)
            if heuristic_initial:
                self.fc.apply(hun_weights)
            else:
                self.fc.apply(init_weights)
            self.heuristic = nn.Linear(model_resnet.fc.in_features, class_num, bias=False)
            self.heuristic.apply(init_weights)
            self.heuristic1 = nn.Linear(model_resnet.fc.in_features, class_num, bias=False)
            self.heuristic1.apply(one_weights)
            self.heuristic2 = nn.Linear(model_resnet.fc.in_features, class_num, bias=False)
            self.heuristic2.apply(two_weights)
            # self.heuristic3 = nn.Linear(model_resnet.fc.in_features, class_num)
            # self.heuristic3.apply(three_weights)
            # self.heuristic4 = nn.Linear(model_resnet.fc.in_features, class_num)
            # self.heuristic4.apply(four_weights)
            self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x, heuristic=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        #
        x = F.normalize(x, dim=1)
        #
        if self.heuristic_num == 1:
            geuristic = self.heuristic(x)
        elif self.heuristic_num == 2:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now_all = torch.cat((now1, now2), 0).reshape(self.heuristic_num, -1, now1.shape[1])
            geuristic = now1 + now2
        elif self.heuristic_num == 3:
            now1 = self.heuristic(x) / self.temp
            now2 = self.heuristic1(x) / self.temp
            now3 = self.heuristic2(x) / self.temp
            now_all = torch.cat((now1, now2, now3), 0).reshape(self.heuristic_num, -1, now1.shape[1])
            geuristic = (now1 + now2 + now3)
        elif self.heuristic_num == 4:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now4 = self.heuristic3(x)
            geuristic = (now1 + now2 + now3 + now4)
        elif self.heuristic_num == 5:
            now1 = self.heuristic(x)
            now2 = self.heuristic1(x)
            now3 = self.heuristic2(x)
            now4 = self.heuristic3(x)
            now5 = self.heuristic4(x)
            geuristic = (now1 + now2 + now3 + now4 + now5)
        y = self.fc(x) / self.temp
        if heuristic:
            y = y - geuristic
        return x, y, geuristic

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'feature_layers' in name or 'conv' in name or 'bn' in name or 'layer' in name:
                    optim_param.append({'params': param, 'lr': lr})
                # print('GVB Classifier {} will be optimized, lr {}'.format(name, lr))
                elif 'fc' in name or 'heuristic' in name:
                    optim_param.append({'params': param, 'lr': lr * 10})
                else:
                    raise RuntimeError('wrong layer name {}'.format(name))
            else:
                print('{} will be ignored'.format(name))
        return optim_param
