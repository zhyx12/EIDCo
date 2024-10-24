from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Function
import math
from fastda.models import MODELS
import torch.distributions as dist
import torch.nn.init as init


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
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
        print('1')
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        print('2')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)
        print('3')


class GradientReverse(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)


class GradientZoom(torch.autograd.Function):
    scale = 1.0

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output


def grad_reverse(x, lambd=1.0):
    GradientReverse.scale = lambd
    return GradientReverse.apply(x)


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input_size)

    return output


@MODELS.register_module()
class AlexNetBase(nn.Module):
    def __init__(self, pretrained=True):
        super(AlexNetBase, self).__init__()
        model_alexnet = models.alexnet(pretrained=pretrained)
        self.features = nn.Sequential(*list(model_alexnet.features._modules.values())[:])
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.__in_features = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    optim_param.append({'params': param, 'lr': lr * 10, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr * 10))
                else:
                    optim_param.append({'params': param, 'lr': lr, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))

        return optim_param


@MODELS.register_module()
class AlexNetBaseWithFC1(AlexNetBase):
    def __init__(self, pretrained=True):
        super(AlexNetBaseWithFC1, self).__init__(pretrained=pretrained)
        self.classifier_fc1 = nn.Linear(4096, 512)

    def forward(self, x, normalize=True):
        x = super(AlexNetBaseWithFC1, self).forward(x)
        x = self.classifier_fc1(x)
        if normalize:
            x = F.normalize(x)
        return x


@MODELS.register_module()
class VGGBase(nn.Module):
    def __init__(self, pretrained=True, no_pool=False):
        super(VGGBase, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.classifier = nn.Sequential(*list(vgg16.classifier._modules.values())[:-1])
        self.features = nn.Sequential(*list(vgg16.features._modules.values())[:])
        self.s = nn.Parameter(torch.FloatTensor([10]))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 7 * 7 * 512)
        x = self.classifier(x)
        return x

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    optim_param.append({'params': param, 'lr': lr * 10, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr * 10))
                else:
                    optim_param.append({'params': param, 'lr': lr, 'weight_decay': 0.0005})
                    print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))

        return optim_param

@MODELS.register_module()
class VGGBaseWithFC1(VGGBase):
    def __init__(self, pretrained=True):
        super(VGGBaseWithFC1, self).__init__(pretrained=pretrained)
        self.classifier_fc1 = nn.Linear(4096, 512)

    def forward(self, x, normalize=True):
        x = super(VGGBaseWithFC1, self).forward(x)
        x = self.classifier_fc1(x)
        if normalize:
            x = F.normalize(x)
        return x

@MODELS.register_module()
class Classifier_shallow(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Classifier_shallow, self).__init__()
        self.fc = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        weights_init(self)

    def forward(self, x, reverse=False, eta=0.1):
        # x = self.bn(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc(x) / self.temp
        return x, x_out


@MODELS.register_module()
class Classifier_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Classifier_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        weights_init(self)

    def forward(self, x, gt=None, reverse=False, eta=0.1, normalize=True):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)  # reverse 和 normalize可以交换
        x_out = self.fc2(x) / self.temp
        return x_out


@MODELS.register_module()
class Classifier_deep_normalized_weights(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Classifier_deep_normalized_weights, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        fc_param = torch.zeros((num_class, 512))
        init.kaiming_uniform_(fc_param, a=math.sqrt(5))
        self.register_parameter('fc2', param=nn.Parameter(data=fc_param, requires_grad=True))
        self.num_class = num_class
        self.temp = temp
        weights_init(self)

    def forward(self, x, gt=None, reverse=False, eta=0.1, normalize=True):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)  # reverse 和 normalize可以交换
        x_out = x.mm(self.fc2.t()) / self.temp
        return x_out


@MODELS.register_module()
class Classifier_deep_without_fc1(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Classifier_deep_without_fc1, self).__init__()
        self.fc2 = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        weights_init(self)

    def forward(self, x, gt=None, reverse=False, eta=0.1, normalize=False):
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                optim_param.append({'params': param, 'lr': lr, 'weight_decay': 0.0005})
                print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module()
class Classifier_deep_multiview(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, temp_multiview=0.05):
        super(Classifier_deep_multiview, self).__init__()
        #
        weights = torch.zeros(num_class, 512)
        torch.nn.init.kaiming_uniform_(weights, a=math.sqrt(5))
        self.register_parameter('fc',
                                nn.Parameter(data=weights.clone(), requires_grad=True))
        self.num_class = num_class
        self.temp = temp
        self.temp_multiview = temp_multiview
        weights_init(self)
        self.split_num = 8
        self.split_dim = int(512 / self.split_num)

    def forward(self, x, gt=None, reverse=False, eta=0.1, multiview=False):
        if reverse:
            x = grad_reverse(x, eta)
        if not multiview:
            x = F.normalize(x)
            x_out = F.linear(x, weight=self.fc) / self.temp
            if gt is not None:
                return x, x_out, F.cross_entropy(x_out, gt)
            else:
                return x, x_out
        else:
            x_out = []
            start_dim_ind = 0
            end_dim_ind = self.split_dim
            for i in range(self.split_num):
                tmp_x = x[:, start_dim_ind:end_dim_ind]
                tmp_x = F.normalize(tmp_x)
                tmp_out = F.linear(tmp_x, weight=self.fc[:, start_dim_ind:end_dim_ind]) / self.temp_multiview
                x_out.append(tmp_out.unsqueeze(0))
                start_dim_ind += self.split_dim
                end_dim_ind += self.split_dim
            return x_out


@MODELS.register_module(name='barlowtwins_bn')
class BarlowTwinsBN(nn.Module):
    def __init__(self, dim):
        super(BarlowTwinsBN, self).__init__()
        self.bn = nn.BatchNorm1d(num_features=dim, affine=False)

    def forward(self, x):
        return self.bn(x)


@MODELS.register_module(name='barlowtwins_bn_with_predictor')
class BarlowTwinsBNWithPredictor(nn.Module):
    def __init__(self, dim):
        super(BarlowTwinsBNWithPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512, bias=False),
        )
        self.bn = nn.BatchNorm1d(num_features=dim, affine=False)

    def forward(self, x):
        x = self.predictor(x)
        return self.bn(x)


@MODELS.register_module(name='adversarial_negatives')
class Adversary_Negatives(nn.Module):
    def __init__(self, bank_size, dim, ):
        super(Adversary_Negatives, self).__init__()
        self.register_parameter("W", nn.Parameter(data=torch.randn(dim, bank_size), requires_grad=True))

    def forward(self, q, init_mem=False):
        memory_bank = self.W
        memory_bank = nn.functional.normalize(memory_bank, dim=0)
        logit = torch.einsum('nc,ck->nk', [q, memory_bank])
        return memory_bank, self.W, logit


@MODELS.register_module(name='stochastic_classifier')
class Stochastic_Classifier(nn.Module):
    def __init__(self, inc=512, num_class=64, temp=0.05):
        super(Stochastic_Classifier, self).__init__()
        mean_vec = torch.zeros((inc, num_class))
        std_vec = torch.zeros((inc, num_class))
        mean_vec.data.normal_(0, 0.01)
        std_vec.data.normal_(0, 0.01)
        self.register_parameter('mean',
                                nn.Parameter(data=mean_vec.clone(), requires_grad=True))
        self.register_parameter('std', nn.Parameter(data=std_vec.clone(), requires_grad=True))
        self.sampler = dist.Normal(loc=torch.zeros((inc, num_class)), scale=torch.ones((inc, num_class)))
        self.num_class = num_class
        self.temp = temp
        #

    def forward(self, x, gt=None, reverse=False, eta=0.1):
        if self.training:
            # 生成随机分类器
            tmp_fc = self.sampler.sample().cuda() * self.std + self.mean
            if reverse:
                x = grad_reverse(x, eta)
            x_out = x.mm(tmp_fc) / self.temp
            if gt is not None:
                return x, x_out, F.cross_entropy(x_out, gt)
            else:
                return x, x_out
        else:
            x_out = x.mm(self.mean) / self.temp
            return x, x_out

    def get_classifier_weights(self, normalize=True):
        fc_weights = self.sampler.sample().cuda() * self.std + self.mean
        if normalize:
            fc_weights = F.normalize(fc_weights, dim=0)
        return fc_weights


@MODELS.register_module(name='adco_predictor')
class AdCOPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, output_dim=512):
        super(AdCOPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.predictor(x)


@MODELS.register_module(name='embedding_network')
class Embedding_Network(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super(Embedding_Network, self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, 2048),
                                nn.ReLU(inplace=True),
                                nn.Linear(2048, output_dim))

    def forward(self, input, reverse=False, eta=1.0):
        if reverse:
            input = grad_reverse(input, eta)
        return self.fc(input)


@MODELS.register_module(name='linear_embedding_network')
class Linear_Embedding_Network(nn.Module):
    def __init__(self, input_dim=512, output_dim=128):
        super(Linear_Embedding_Network, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.fc(input)


@MODELS.register_module()
class Classifier_deep_without_fc1_transfer(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, fix_weight_transfer=False, lambda_weight_transfer_lr=0.1,
                 init_type_for_non_diag='zero', init_std=0.005, force_detach=True):
        super(Classifier_deep_without_fc1_transfer, self).__init__()
        fc_param = torch.zeros((num_class, inc))
        init.kaiming_uniform_(fc_param, a=math.sqrt(5))
        self.register_parameter('fc2', param=nn.Parameter(data=fc_param, requires_grad=True))
        #
        weight_transfer_param = torch.eye(inc)
        if init_type_for_non_diag == 'zero':
            pass
        elif init_type_for_non_diag == 'normal':
            tmp_weights = torch.ones((inc, inc))
            init.normal_(tmp_weights, mean=0, std=init_std)
            weight_transfer_param += tmp_weights * weight_transfer_param
        else:
            raise RuntimeError('wrong type of init')
        self.register_parameter('transfer_weight',
                                param=nn.Parameter(data=weight_transfer_param, requires_grad=not fix_weight_transfer))
        self.num_class = num_class
        self.temp = temp
        self.lambda_weight_transfer_lr = lambda_weight_transfer_lr
        if fix_weight_transfer:
            if not force_detach:
                self.detach = False
            else:
                self.detach = True  # 和之前的“baseline”兼容
        else:
            self.detach = True

    def forward(self, x, target=False, gt=None, reverse=False, eta=0.1, normalize=False):
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)
        #
        if target:
            if not self.detach:
                tmp_weights = self.fc2
            else:
                tmp_weights = self.fc2.detach().mm(self.transfer_weight)
            x_out = x.mm(tmp_weights.t()) / self.temp
        else:
            x_out = x.mm(self.fc2.t()) / self.temp
        if gt is not None:
            return x_out, F.cross_entropy(x_out, gt)
        else:
            return x_out

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if 'fc' in name:
                optim_param.append({'params': param, 'lr': lr})
            elif 'transfer_weight' in name:
                optim_param.append({'params': param, 'lr': lr * self.lambda_weight_transfer_lr, 'weight_decay': 0.0})
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module()
class Classifier_deep_transfer(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, fix_weight_transfer=False, lambda_weight_transfer_lr=0.1):
        super(Classifier_deep_transfer, self).__init__()
        #
        fc_param = torch.zeros((512, inc))
        init.kaiming_uniform_(fc_param, a=math.sqrt(5))
        self.register_parameter('fc1', param=nn.Parameter(data=fc_param, requires_grad=True))
        weight_transfer_param = torch.eye(inc)
        self.register_parameter('transfer_weight_1',
                                param=nn.Parameter(data=weight_transfer_param, requires_grad=not fix_weight_transfer))
        #
        fc_param = torch.zeros((num_class, 512))
        init.kaiming_uniform_(fc_param, a=math.sqrt(5))
        self.register_parameter('fc2', param=nn.Parameter(data=fc_param, requires_grad=True))
        #
        weight_transfer_param = torch.eye(inc)
        self.register_parameter('transfer_weight',
                                param=nn.Parameter(data=weight_transfer_param, requires_grad=not fix_weight_transfer))
        self.num_class = num_class
        self.temp = temp
        self.lambda_weight_transfer_lr = lambda_weight_transfer_lr

    def forward(self, x, target=False, gt=None, reverse=False, eta=0.1, normalize=True):
        if target:
            tmp_weights = self.fc1.detach().mm(self.transfer_weight_1)
            x = x.mm(tmp_weights.t())
        else:
            x = x.mm(self.fc1.t())
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)
        #
        if target:
            tmp_weights = self.fc2.detach().mm(self.transfer_weight)
            x_out = x.mm(tmp_weights.t()) / self.temp
        else:
            x_out = x.mm(self.fc2.t()) / self.temp
        if gt is not None:
            return x_out, F.cross_entropy(x_out, gt)
        else:
            return x_out

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if 'fc' in name:
                optim_param.append({'params': param, 'lr': lr})
            elif 'transfer_weight_1' in name:
                optim_param.append({'params': param, 'lr': lr * self.lambda_weight_transfer_lr, 'weight_decay': 0.0})
            elif 'transfer_weight' in name:
                optim_param.append({'params': param, 'lr': lr * self.lambda_weight_transfer_lr, 'weight_decay': 0.0})
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module()
class Classifier_deep_without_fc1_1D_transfer(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, fix_weight_transfer=False, lambda_weight_transfer_lr=1.0):
        super(Classifier_deep_without_fc1_1D_transfer, self).__init__()
        fc_param = torch.zeros((num_class, inc))
        init.kaiming_uniform_(fc_param, a=math.sqrt(5))
        self.register_parameter('fc2', param=nn.Parameter(data=fc_param, requires_grad=True))
        #
        weight_transfer_param = torch.ones((inc,))
        self.register_parameter('transfer_weight',
                                param=nn.Parameter(data=weight_transfer_param, requires_grad=not fix_weight_transfer))
        self.num_class = num_class
        self.temp = temp
        self.lambda_weight_transfer_lr = lambda_weight_transfer_lr

    def forward(self, x, target=False, gt=None, reverse=False, eta=0.1, normalize=False):
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)
        #
        if target:
            tmp_weights = self.fc2.detach()
            x_out = (x * self.transfer_weight).mm(tmp_weights.t()) / self.temp
        else:
            x_out = x.mm(self.fc2.t()) / self.temp
        if gt is not None:
            return x, x_out, F.cross_entropy(x_out, gt)
        else:
            return x, x_out

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if 'fc' in name:
                optim_param.append({'params': param, 'lr': lr})
            elif 'transfer_weight' in name:
                optim_param.append({'params': param, 'lr': lr * self.lambda_weight_transfer_lr, 'weight_decay': 0.0})
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module()
class Classifier_deep_without_fc1_same_transfer(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05, fix_weight_transfer=False, lambda_weight_transfer_lr=1.0):
        super(Classifier_deep_without_fc1_same_transfer, self).__init__()
        fc_param = torch.zeros((num_class, inc))
        init.kaiming_uniform_(fc_param, a=math.sqrt(5))
        self.register_parameter('fc2', param=nn.Parameter(data=fc_param, requires_grad=True))
        #
        weight_transfer_param = torch.ones((num_class, inc))
        self.register_parameter('transfer_weight',
                                param=nn.Parameter(data=weight_transfer_param, requires_grad=not fix_weight_transfer))
        self.num_class = num_class
        self.temp = temp
        self.lambda_weight_transfer_lr = lambda_weight_transfer_lr

    def forward(self, x, target=False, gt=None, reverse=False, eta=0.1, normalize=False):
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)
        #
        if target:
            tmp_weights = self.fc2.detach() * self.transfer_weight
            x_out = x.mm(tmp_weights.t()) / self.temp
        else:
            x_out = x.mm(self.fc2.t()) / self.temp
        if gt is not None:
            return x, x_out, F.cross_entropy(x_out, gt)
        else:
            return x, x_out

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if 'fc' in name:
                optim_param.append({'params': param, 'lr': lr})
            elif 'transfer_weight' in name:
                optim_param.append({'params': param, 'lr': lr * self.lambda_weight_transfer_lr, 'weight_decay': 0.0})
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module()
class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature=512, hidden_size=1024):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)

    def forward(self, x, reverse=False, eta=1.0):
        if reverse:
            x = grad_reverse(x, eta)
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y


@MODELS.register_module(name='stochastic_classifier_with_transfer')
class Stochastic_Classifier_with_Transfer(nn.Module):
    def __init__(self, inc=512, num_class=64, temp=0.05, fix_weight_transfer=False, lambda_weight_transfer_lr=0.1,
                 init_type_for_non_diag='zero', init_std=0.005, std=0.001):
        super(Stochastic_Classifier_with_Transfer, self).__init__()
        mean_vec = torch.zeros((num_class, inc))
        std_vec = torch.zeros((num_class, inc))
        init.kaiming_uniform_(mean_vec, a=math.sqrt(5))
        std_vec.data.normal_(0, std)
        self.register_parameter('mean',
                                nn.Parameter(data=mean_vec.clone(), requires_grad=True))
        self.register_parameter('std', nn.Parameter(data=std_vec.clone(), requires_grad=True))
        self.sampler = dist.Normal(loc=torch.zeros((num_class, inc)), scale=torch.ones((num_class, inc)))
        self.num_class = num_class
        self.temp = temp
        self.lambda_weight_transfer_lr = lambda_weight_transfer_lr
        #
        weight_transfer_param = torch.eye(inc)
        if init_type_for_non_diag == 'zero':
            pass
        elif init_type_for_non_diag == 'normal':
            tmp_weights = torch.ones((inc, inc))
            init.normal_(tmp_weights, mean=0, std=init_std)
            weight_transfer_param += tmp_weights * weight_transfer_param
        else:
            raise RuntimeError('wrong type of init')
        self.register_parameter('transfer_weight',
                                param=nn.Parameter(data=weight_transfer_param, requires_grad=not fix_weight_transfer))
        #

    def forward(self, x, gt=None, reverse=False, eta=0.1, target=False, normalize=False):
        if reverse:
            x = grad_reverse(x, eta)
        if normalize:
            x = F.normalize(x)
        #
        if self.training:
            # 生成随机分类器
            tmp_fc = self.sampler.sample().cuda() * self.std + self.mean
            if target:
                tmp_weights = tmp_fc.detach().mm(self.transfer_weight)
                x_out = x.mm(tmp_weights.t()) / self.temp
            else:
                x_out = x.mm(tmp_fc.t()) / self.temp
            if gt is not None:
                return x, x_out, F.cross_entropy(x_out, gt)
            else:
                return x, x_out
        else:
            if target:
                tmp_weights = self.mean.detach().mm(self.transfer_weight)
            else:
                tmp_weights = self.mean
            x_out = x.mm(tmp_weights.t()) / self.temp
            return x, x_out

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if 'mean' in name or 'std' in name:
                optim_param.append({'params': param, 'lr': lr})
            elif 'transfer_weight' in name:
                optim_param.append({'params': param, 'lr': lr * self.lambda_weight_transfer_lr, 'weight_decay': 0.0})
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module(name='random_layer')
class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=(), output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        for i in range(self.input_num):
            self.register_buffer('random_matrix_{}'.format(i), torch.randn(input_dim_list[i], output_dim))

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], getattr(self, 'random_matrix_{}'.format(i))) for i in
                       range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor


@MODELS.register_module(name='lambda_net')
class LambdaNet(nn.Module):
    def __init__(self, input_dim, ):
        super(LambdaNet, self).__init__()
        self.lambda_net = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x1, x2):
        input_1 = x1
        input_2 = x2
        input_for_lambda = torch.cat((input_1, input_2), dim=1)
        lam = self.lambda_net(input_for_lambda)  # shape(B,1)
        feature = F.normalize(x1 * lam + x2 * (1 - lam), dim=1)
        return lam, feature


@MODELS.register_module(name='low_feature_discriminator')
class LowFeatureDiscriminator(nn.Module):
    def __init__(self, input_dim, ):
        super(LowFeatureDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, reverse_grad=False):
        if reverse_grad:
            x = grad_reverse(x)
        x = self.discriminator(x)
        return x


@MODELS.register_module(name='multi_lambda_net')
class MultiLambdaNet(nn.Module):
    def __init__(self, input_dim, ):
        super(MultiLambdaNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=1024),
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=1024),
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=1024),
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=1024),
        )
        self.lambda_head = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        encode_x = self.encoder(x)
        b, n, c = encode_x.shape
        encode_x = encode_x.view(b * n, c)
        lambda_x = self.lambda_head(encode_x).view(b, n)
        lam = F.softmax(lambda_x, dim=1)
        feature = F.normalize(torch.sum(lam.unsqueeze(2) * x, dim=1), dim=1)
        return lam, feature


@MODELS.register_module()
class GVBClassifier(nn.Module):
    def __init__(self, num_class=64, inc=512, temp=0.05):
        super(GVBClassifier, self).__init__()
        self.fc2 = nn.Linear(inc, num_class, bias=False)
        self.bridge = nn.Linear(inc, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp
        self.apply(init_weights)

    def forward(self, x, gvbg=True, normalize=False, reverse=False, eta=1.0):
        if normalize:
            x = F.normalize(x)
        bridge = self.bridge(x) / self.temp
        x_out = self.fc2(x) / self.temp
        if gvbg:
            x_out = x_out - bridge
        return x_out, bridge

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                optim_param.append({'params': param, 'lr': lr})
                print('GVB Classifier {} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))
        return optim_param


@MODELS.register_module()
class OrigGVBClassifier(nn.Module):
    def __init__(self, num_class=64, inc=512):
        super(OrigGVBClassifier, self).__init__()
        self.fc2 = nn.Linear(inc, num_class)
        self.bridge = nn.Linear(inc, num_class)
        self.num_class = num_class
        # init_weights(self)
        self.apply(init_weights)

    def forward(self, x, gvbg=True, normalize=False):
        if normalize:
            x = F.normalize(x)
        bridge = self.bridge(x)
        x_out = self.fc2(x)
        if gvbg:
            x_out = x_out - bridge
        return x_out, bridge

    def optim_parameters(self, lr):
        optim_param = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                optim_param.append({'params': param, 'lr': lr})
                print('{} will be optimized, lr {}'.format(name, lr))
            else:
                print('{} will be ignored'.format(name))
        return optim_param




if __name__ == '__main__':
    model = Classifier_deep_multiview().cuda()
    input = torch.ones(10, 512).cuda()
    output = model(input, multiview=True)
    print('output')
