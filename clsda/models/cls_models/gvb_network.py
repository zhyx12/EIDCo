import numpy as np
import torch.nn as nn
from torchvision import models
import torch
from fastda.models import MODELS
from .basenet import grad_reverse
import torch.nn.functional as F
from .resnet import ResNetWithNorm


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50,
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


@MODELS.register_module()
class GVBResNetFc(nn.Module):
    def __init__(self, resnet_name, class_num=1000):
        super(GVBResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #
        self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
        self.fc.apply(init_weights)
        self.gvbg = nn.Linear(model_resnet.fc.in_features, class_num)
        self.gvbg.apply(init_weights)
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x, gvbg=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        bridge = self.gvbg(x)
        y = self.fc(x)
        if gvbg:
            y = y - bridge
        return y, bridge

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                          {"params": self.fc.parameters(), "lr": lr * 10},
                          {"params": self.gvbg.parameters(), "lr": lr * 10}]
        return parameter_list


@MODELS.register_module()
class GVBResNetCosine(nn.Module):
    def __init__(self, resnet_name, class_num=1000, temp=0.05):
        super(GVBResNetCosine, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #
        self.fc = nn.Linear(model_resnet.fc.in_features, class_num, bias=False)
        self.fc.apply(init_weights)
        self.gvbg = nn.Linear(model_resnet.fc.in_features, class_num, bias=False)
        self.gvbg.apply(init_weights)
        self.__in_features = model_resnet.fc.in_features
        self.temp = temp

    def forward(self, x, gvbg=True, normalize=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if normalize:
            x = F.normalize(x, dim=1)
        bridge = self.gvbg(x) / self.temp
        y = self.fc(x) / self.temp
        if gvbg:
            y = y - bridge
        return y, bridge

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                          {"params": self.fc.parameters(), "lr": lr * 10},
                          {"params": self.gvbg.parameters(), "lr": lr * 10}]
        return parameter_list


@MODELS.register_module()
class GVBResNetBase(nn.Module):
    def __init__(self, resnet_name, class_num=1000):
        super(GVBResNetBase, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #

    def forward(self, x, gvbg=True, normalize=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if normalize:
            x = F.normalize(x, dim=1)
        return x

    def output_num(self):
        return 2048

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                          ]
        return parameter_list


@MODELS.register_module()
class GVBResNetFC1(nn.Module):
    def __init__(self, resnet_name, fc_dim=512, class_num=1000):
        super(GVBResNetFC1, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        #
        expansion = 4 if resnet_name in ['ResNet50', "ResNet101", 'ResNet152'] else 1
        self.classifier_fc1 = nn.Linear(512 * expansion, fc_dim)

    def forward(self, x, gvbg=True, normalize=True):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier_fc1(x)
        if normalize:
            x = F.normalize(x, dim=1)
        return x

    def output_num(self):
        return 2048

    def optim_parameters(self, lr):
        parameter_list = [{"params": self.feature_layers.parameters(), "lr": lr},
                          {"params": self.classifier_fc1.parameters(), "lr": lr * 10}
                          ]
        return parameter_list


@MODELS.register_module()
class GVBResNetConcat(nn.Module):
    def __init__(self, class_num=1000):
        super(GVBResNetConcat, self).__init__()
        from .basenet import GVBClassifier
        # self.backbone = GVBResNetBase(resnet_name='ResNet50')
        self.backbone = ResNetWithNorm(depth=50)
        self.classifier = GVBClassifier(num_class=65, inc=2048)
        #

    def forward(self, x, gvbg=True):
        x = self.backbone(x)
        # x = F.normalize(x,dim=1)
        x = self.classifier(x)
        return x

    def output_num(self):
        return 2048

    def optim_parameters(self, lr):
        parameter_list = []
        parameter_list.extend(self.backbone.optim_parameters(lr))
        parameter_list.extend(self.classifier.optim_parameters(lr * 10))
        return parameter_list


@MODELS.register_module()
class GVBAdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size=1024, multi=1):
        super(GVBAdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, multi)
        self.gvbd = nn.Linear(hidden_size, multi)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x, iteration):
        coeff = calc_coeff(iteration)
        x = grad_reverse(x, lambd=coeff)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        z = self.gvbd(x)
        return y, z

    def output_num(self):
        return 1

    def optim_parameters(self, lr):
        # TODO: 训练参数
        return [{"params": self.parameters(), "lr": 10 * lr, 'weight_decay': 0.001}]


class Myloss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Myloss, self).__init__()
        self.epsilon = epsilon
        return

    def forward(self, input_, label, weight):
        entropy = - label * torch.log(input_ + self.epsilon) - (1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def GVBLoss(input_list, ad_net, coeff=None, myloss=Myloss(), GVBD=False, iteration=None):
    softmax_output = input_list[0]
    focals = input_list[1].reshape(-1)
    ad_out, fc_out = ad_net(softmax_output, iteration=iteration)
    if GVBD:
        ad_out = nn.Sigmoid()(ad_out - fc_out)
    else:
        ad_out = nn.Sigmoid()(ad_out)
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().to(softmax_output.device)

    x = softmax_output
    entropy = Entropy(x)
    entropy = grad_reverse(entropy, lambd=coeff)
    entropy = torch.exp(-entropy)
    mean_entropy = torch.mean(entropy)
    gvbg = torch.mean(torch.abs(focals))
    gvbd = torch.mean(torch.abs(fc_out))

    source_mask = torch.ones_like(entropy)
    source_mask[softmax_output.size(0) // 2:] = 0
    source_weight = entropy * source_mask
    target_mask = torch.ones_like(entropy)
    target_mask[0:softmax_output.size(0) // 2] = 0
    target_weight = entropy * target_mask
    weight = source_weight / torch.sum(source_weight).detach().item() + \
             target_weight / torch.sum(target_weight).detach().item()
    return myloss(ad_out, dc_target, weight.view(-1, 1)), mean_entropy, gvbg, gvbd
