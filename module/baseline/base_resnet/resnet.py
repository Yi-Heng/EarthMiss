from functools import partial
import torch.nn as nn
from torch.utils import checkpoint as cp
import torch
from ever.core import registry, logger
from ever.interface import ERModule
from ._resnets import resnet101
from ._resnets import resnet18
from ._resnets import resnet34
from ._resnets import resnet50
import torch.nn.functional as F
from ever.util import param_util

_logger = logger.get_logger()
__all__ = ['make_layer',
           'ResNetEncoder']

_RESNET_FUNCTIONS = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
}



def make_layer(block, in_channel, basic_out_channel, blocks, stride=1, dilation=1,wt = 0):
    downsample = None
    if stride != 1 or in_channel != basic_out_channel * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(in_channel, basic_out_channel * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(basic_out_channel * block.expansion),
        )

    layers = []
    layers.append(block(in_channel, basic_out_channel, stride, dilation, downsample))
    in_channel = basic_out_channel * block.expansion
    for i in range(1, blocks):
        layers.append(block(in_channel, basic_out_channel, dilation=dilation))

    return nn.Sequential(*layers)


class ResNetEncoder(ERModule):
    def __init__(self,
                 config):
        super(ResNetEncoder, self).__init__(config)
        if all([self.config.output_stride != 16,
                self.config.output_stride != 32,
                self.config.output_stride != 8]):
            raise ValueError('output_stride must be 8, 16 or 32.')

        resnet_fn = _RESNET_FUNCTIONS.get(self.config.resnet_type)
        if resnet_fn is None:
            raise ValueError(f'Unsupported resnet_type: {self.config.resnet_type}. '
                           f'Supported types: {list(_RESNET_FUNCTIONS.keys())}')
        self.resnet = resnet_fn(pretrained=self.config.pretrained,
                               norm_layer=self.config.norm_layer,
                               wt_layer=self.config.wt_layer)
        _logger.info('ResNetEncoder: pretrained = {}'.format(self.config.pretrained))
        self.resnet._modules.pop('fc')
        self.set_in_channels(self.config.in_channels)
        if not self.config.batchnorm_trainable:
            self._frozen_res_bn()

        self._freeze_at(at=self.config.freeze_at)
        if self.config.output_stride == 16:
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=2))
        elif self.config.output_stride == 8:
            self.resnet.layer3.apply(partial(self._nostride_dilate, dilate=2))
            self.resnet.layer4.apply(partial(self._nostride_dilate, dilate=4))
        self.applt_wt = True if 2 in self.config.wt_layer else False
     
        if self.config.wt_layer[2] == 1:
            self.resnet.bn1 = nn.InstanceNorm2d(64, affine=True)
            self.resnet.relu = nn.ReLU(inplace=True)
        if self.config.wt_layer[2] == 2:
            self.resnet.bn1 = nn.InstanceNorm2d(64, affine=False)
            self.resnet.relu = nn.ReLU(inplace=False)

    def patch_first_conv(self,model, new_in_channels, default_in_channels=3, pretrained=True):
        """Change first convolution layer input channels.
        In case:
            in_channels == 1 or in_channels == 2 -> reuse original weights
            in_channels > 3 -> make random kaiming normal initialization
        """
    # get first conv
        for module in model.modules():
            if isinstance(module, nn.Conv2d) and module.in_channels == default_in_channels:
                break

        weight = module.weight.detach()
        module.in_channels = new_in_channels

        if not pretrained:
            module.weight = nn.parameter.Parameter(
                torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)
            )
            module.reset_parameters()
        #卷积的尺寸是 输出通道、输入通道、高、宽,keepdim是保持四个维度，但输入通道会变成1 
        elif new_in_channels == 1:
            new_weight = weight.sum(1, keepdim=True)
            module.weight = nn.parameter.Parameter(new_weight)

        else:
            new_weight = torch.Tensor(module.out_channels, new_in_channels // module.groups, *module.kernel_size)

            for i in range(new_in_channels):
                new_weight[:, i] = weight[:, i % default_in_channels]

            new_weight = new_weight * (default_in_channels / new_in_channels)
            module.weight = nn.parameter.Parameter(new_weight)
    def set_in_channels(self, in_channels, pretrained=True):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        self.patch_first_conv(model=self.resnet, new_in_channels=in_channels, pretrained=pretrained)
    @property
    def layer1(self):
        return self.resnet.layer1

    @layer1.setter
    def layer1(self, value):
        del self.resnet.layer1
        self.resnet.layer1 = value

    @property
    def layer2(self):
        return self.resnet.layer2

    @layer2.setter
    def layer2(self, value):
        del self.resnet.layer2
        self.resnet.layer2 = value

    @property
    def layer3(self):
        return self.resnet.layer3

    @layer3.setter
    def layer3(self, value):
        del self.resnet.layer3
        self.resnet.layer3 = value

    @property
    def layer4(self):
        return self.resnet.layer4

    @layer4.setter
    def layer4(self, value):
        del self.resnet.layer4
        self.resnet.layer4 = value

    def _frozen_res_bn(self):
        _logger.info('ResNetEncoder: freeze all BN layers')
        param_util.freeze_modules(self.resnet, nn.modules.batchnorm._BatchNorm)
        for m in self.resnet.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()

    def _freeze_at(self, at=2):
        if at >= 1:
            param_util.freeze_params(self.resnet.conv1)
            param_util.freeze_params(self.resnet.bn1)
        if at >= 2:
            param_util.freeze_params(self.resnet.layer1)
        if at >= 3:
            param_util.freeze_params(self.resnet.layer2)
        if at >= 4:
            param_util.freeze_params(self.resnet.layer3)
        if at >= 5:
            param_util.freeze_params(self.resnet.layer4)

    @staticmethod
    def get_function(module):
        def _function(x):
            y = module(x)
            return y

        return _function

    def forward(self, inputs):
        x = inputs

        w_arr = []
        if self.config.wt_layer[2] == 2:
            x,w1 = self.resnet.stem_forward(x)
            w_arr.append(w1)
        else:
            x = self.resnet.stem_forward(x)
        x = self.resnet.maxpool(x)

        # os 4, #layers/outdim: 18,34/64; 50,101,152/256
        if self.config.with_cp[0] and x.requires_grad:
            c2 = cp.checkpoint(self.get_function(self.resnet.layer1), x)
        else:
            if self.config.wt_layer[3] == 2:
                c2,w2 = self.resnet.layer1(x)
                w_arr.append(w2)
            else:
                c2 = self.resnet.layer1(x)
        # os 8, #layers/outdim: 18,34/128; 50,101,152/512
        if self.config.with_cp[1] and c2.requires_grad:
            c3 = cp.checkpoint(self.get_function(self.resnet.layer2), c2)
        else:
            if self.config.wt_layer[4] == 2:
                c3,w3 = self.resnet.layer2(c2)
                w_arr.append(w3)
            else:
                c3 = self.resnet.layer2(c2)
        # os 16, #layers/outdim: 18,34/256; 50,101,152/1024
        if self.config.with_cp[2] and c3.requires_grad:
            c4 = cp.checkpoint(self.get_function(self.resnet.layer3), c3)
        else:
            if self.config.wt_layer[5] == 2:
                c4,w4 = self.resnet.layer3(c3)
                w_arr.append(w4)
            else:
                c4 = self.resnet.layer3(c3)
        # os 32, #layers/outdim: 18,34/512; 50,101,152/2048
        if self.config.include_conv5:
            if self.config.with_cp[3] and c4.requires_grad:
                c5 = cp.checkpoint(self.get_function(self.resnet.layer4), c4)
            else:
                if self.config.wt_layer[6] == 2:
                    c5,w5 = self.resnet.layer4(c4)
                    w_arr.append(w5)
                else:
                    c5 = self.resnet.layer4(c4)
            if w_arr:
                return [c2, c3, c4, c5],w_arr
            else:
                return [c2, c3, c4, c5]

        if w_arr:
                return [c2, c3, c4],w_arr
        else:
            return [c2, c3, c4]





    def set_default_config(self):
        self.config.update(dict(
            resnet_type='resnet50',
            include_conv5=True,
            batchnorm_trainable=True,
            pretrained=False,
            freeze_at=0,
            # 16 or 32
            output_stride=32,
            with_cp=(False, False, False, False),
            norm_layer=nn.BatchNorm2d,
            wt_layer = [0,0,0,0,0,0,0]
        ))

    def train(self, mode=True):
        super(ResNetEncoder, self).train(mode)
        self._freeze_at(self.config.freeze_at)
        if mode and not self.config.batchnorm_trainable:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()

    def _nostride_dilate(self, m, dilate):
        # ref:
        # https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/1235deb1d68a8f3ef87d639b95b2b8e3607eea4c/models/models.py#L256
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)


