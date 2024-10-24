import torch
import torch.nn as nn
import torch.nn.functional as F

from .ViT import VT, vit_model
from .grl import WarmStartGradientReverseLayer
from fastda.models import MODELS


@MODELS.register_module(name='vit_grl_basenet')
class ViTgrlBaseNet(nn.Module):
    def __init__(self, base_net='vit_base_patch16_224', use_bottleneck=True, bottleneck_dim=1024, width=1024,
                 args=None):
        super(ViTgrlBaseNet, self).__init__()

        self.base_network = vit_model[base_net](pretrained=True, args=args, VisionTransformerModule=VT)
        self.use_bottleneck = use_bottleneck
        self.grl = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000, auto_step=True)
        if self.use_bottleneck:
            self.bottleneck_layer = [nn.Linear(self.base_network.embed_dim, bottleneck_dim),
                                     nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
            self.bottleneck = nn.Sequential(*self.bottleneck_layer)

        classifier_dim = bottleneck_dim if use_bottleneck else self.base_network.embed_dim
        self.classifier_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.5)]
        self.classifier = nn.Sequential(*self.classifier_layer)

        self.discriminator_layer = [nn.Linear(classifier_dim, width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, 1)]
        self.discriminator = nn.Sequential(*self.discriminator_layer)

        if self.use_bottleneck:
            self.bottleneck[0].weight.data.normal_(0, 0.005)
            self.bottleneck[0].bias.data.fill_(0.1)

        for dep in range(2):
            self.discriminator[dep * 3].weight.data.normal_(0, 0.01)
            self.discriminator[dep * 3].bias.data.fill_(0.0)
        #
        self.classifier[0].weight.data.normal_(0, 0.01)
        self.classifier[0].bias.data.fill_(0.0)

    def forward(self, inputs, normalize=True):
        features = self.base_network.forward_features(inputs)
        if self.use_bottleneck:
            features = self.bottleneck(features)

        feat = self.classifier(features)

        if normalize:
            feat = F.normalize(feat, dim=1)

        if self.training:
            outputs_dc = self.discriminator(self.grl(features))
            return feat, outputs_dc
        else:
            return feat

    def optim_parameters(self, lr):
        optim_param = [
            {"params": self.base_network.parameters(), "lr": lr * 0.1},
            {"params": self.classifier.parameters(), "lr": lr * 1.0},
            {"params": self.discriminator.parameters(), "lr": lr * 1.0}]
        #
        if self.use_bottleneck:
            optim_param.extend([{"params": self.bottleneck.parameters(), "lr": lr * 1.0}])

        return optim_param


@MODELS.register_module(name='vit_classifier')
class VitClassifier(nn.Module):
    def __init__(self, width=1024, class_num=65, ):
        super(VitClassifier, self).__init__()

        self.fc = nn.Linear(width, class_num)
        self.fc.weight.data.normal_(0, 0.01)
        self.fc.bias.data.fill_(0.0)

    def forward(self, inputs):
        return self.fc(inputs)

    def optim_parameters(self, lr):
        optim_param = [
            {"params": self.fc.parameters(), "lr": lr * 1.0}, ]

        return optim_param


@MODELS.register_module(name='vit_cosine_classifier')
class VitCosineClassifier(nn.Module):
    def __init__(self, width=1024, class_num=65, temp=0.05):
        super(VitCosineClassifier, self).__init__()

        self.fc2 = nn.Linear(width, class_num, bias=False)
        self.fc2.weight.data.normal_(0, 0.01)
        # self.fc.bias.data.fill_(0.0)
        self.temp = temp

    def forward(self, inputs, normalize=False, ):
        if normalize:
            inputs = F.normalize(inputs, dim=1)
        return self.fc2(inputs) / self.temp

    def optim_parameters(self, lr):
        optim_param = [
            {"params": self.fc2.parameters(), "lr": lr * 1.0}, ]

        return optim_param
