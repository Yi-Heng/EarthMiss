import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ever.interface import ERModule
from ever import registry
from module.baseline.base_resnet.resnet import ResNetEncoder
from module.baseline.base_metars.mmr import CovMatrix_mmr,instance_whitening_loss
from module.baseline.base import AssymetricDecoder
from ever.module import FPN
from module.loss import SegmentationLoss
from module.baseline.base_metars.model import Fusespfnv2,Concat,Fusespfnv2_res
import os
import logging
from ever.core.builder import make_dataloader
from tqdm import tqdm
import torch.distributed as dist

logger = logging.getLogger(__name__)

@registry.MODEL.register('MetaRS')
class MetaRS(ERModule):
    def __init__(self, config):
        super(MetaRS, self).__init__(config)
        org_wt_layer = self.config.encoder.wt_layer
        self.config.encoder.wt_layer = [0,0,0,0,0,0,0]
        self.config.encoder.in_channels = self.config.sar_in_channels
        self.en_sar = ResNetEncoder(self.config.encoder)
        self.config.encoder.in_channels = self.config.rgb_in_channels
        self.en_rgb = ResNetEncoder(self.config.encoder)

        self.config.encoder.wt_layer = org_wt_layer
        self.config.encoder.in_channels = self.config.sar_in_channels
        self.en_content = ResNetEncoder(self.config.encoder)

        self.Project = Fusespfnv2_res(self.config.fuse_inchan_list)
        self.Fuse = Fusespfnv2(self.config.fuse_inchan_list)
        self.decoder_sar =  nn.Sequential(
            FPN(**self.config.fpn),
            AssymetricDecoder(**self.config.decoder),
            nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.decoder_rgb =  nn.Sequential(
            FPN(**self.config.fpn),
            AssymetricDecoder(**self.config.decoder),
            nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.config.fpn["in_channels_list"] = self.config.fuse_inchan_list
        self.decoder =  nn.Sequential(
            FPN(**self.config.fpn),
            AssymetricDecoder(**self.config.decoder),
            nn.Conv2d(self.config.decoder.out_channels, self.config.num_classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.cls_loss = SegmentationLoss(self.config.loss)
        try:
            self.local_rank=int(os.environ["LOCAL_RANK"])
            if self.local_rank == 0:
                print(self.config)
                print(self.en_content)
        except:
            self.local_rank = None
        self.mse_loss = torch.nn.MSELoss()
        self.Concat = Concat()
        self.add_module('rgb_conv1', nn.Conv2d(self.config.rgb_in_channels,
                                        64, kernel_size=7, stride=2, padding=3,
                                        bias=False))
        self.infer_rgb = self.config.infer_rgb

        self.apply_mmr = 2 in self.config.encoder.wt_layer
        self.eps = 1e-5
        self.cov_matrix_layer = []
        self.cov_type = []
        for i in range(len(self.config.encoder.wt_layer)):
            if self.config.encoder.wt_layer[i] == 2 :
                self.cov_matrix_layer.append(CovMatrix_mmr(dim=self.config.in_channels_list[i], relax_denom=self.config.relax_denom, clusters=self.config.clusters))
                self.cov_type.append(self.config.encoder.wt_layer[i])
        
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))

    def load_state_dict(self, state_dict, strict=True):
        from collections import OrderedDict
        if isinstance(state_dict, OrderedDict):
            state_dict = OrderedDict(state_dict)
        else:
            state_dict = dict(state_dict)
        
        has_global_step = any('global_step' in k for k in state_dict.keys())
        if not has_global_step:
            if state_dict and list(state_dict.keys())[0].startswith('module.'):
                state_dict['module.global_step'] = torch.tensor(0, dtype=torch.long)
            else:
                state_dict['global_step'] = torch.tensor(0, dtype=torch.long)
        super().load_state_dict(state_dict, strict=strict)
    
    def forward(self, x, y=None):
        if self.training:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                if rank == 0:
                    self.global_step += 1
                dist.broadcast(self.global_step, src=0)
            else:
                self.global_step += 1
            
            if self.apply_mmr and self.global_step.item() == self.config.begin_mmr_iter:
                self.conduct_mask_matrix()
        sar = x[:,:self.config.sar_in_channels,:,:]
        rgb = x[:,self.config.sar_in_channels:,:,:]



        sar_w_list,rgb_w_list = [],[]
        sar_content = self.en_content(sar)
        if isinstance(sar_content,tuple):
            if 2 in self.config.encoder.wt_layer:
                sar_content,sar_w_list = sar_content[0],sar_content[1]
        sar_specific = self.en_sar(sar)
        sar_fuse = self.Project(sar_specific,sar_content)
        sar_fuse_f = self.Concat(sar_fuse,sar_content)
        sar_cls_pred = self.decoder_sar(sar_fuse_f)

        org_conv=self.en_content.resnet.conv1
        self.en_content.resnet.conv1=self._modules['rgb_conv1']
        rgb_content = self.en_content(rgb)
        if isinstance(rgb_content,tuple):
            if 2 in self.config.encoder.wt_layer:
                rgb_content,rgb_w_list = rgb_content[0],rgb_content[1]
        rgb_specific = self.en_rgb(rgb)
        rgb_fuse = self.Project(rgb_specific,rgb_content)
        rgb_fuse_f = self.Concat(rgb_fuse,rgb_content)
        rgb_cls_pred = self.decoder_rgb(rgb_fuse_f)
        self.en_content.resnet.conv1=org_conv
        fuse_features = self.Fuse(sar_content,rgb_content)
        fuse_cls_pred = self.decoder(fuse_features)

        if self.training:
            cls_true = y['cls']
            loss_dict = self.cls_loss(sar_cls_pred, cls_true,"sar")
            rgb_loss_dict = self.cls_loss(rgb_cls_pred, cls_true,"rgb")
            fuse_loss_dict = self.cls_loss(fuse_cls_pred, cls_true,"fuse")
            loss_dict.update(rgb_loss_dict)
            loss_dict.update(fuse_loss_dict)

            if "mmr" in self.config.loss:
                if self.apply_mmr and self.global_step.item() > self.config.begin_mmr_iter:
                    loss_dict["mmr_loss"] = self.mmr_Loss(sar_w_list,rgb_w_list).mean()
                    if "mse" in self.config.loss:
                        loss_dict['mse_loss'] = (sum(self.mse_loss(sar_c,rgb_c) for sar_c,rgb_c in zip(sar_content[2:],rgb_content[2:]))/2)*self.config.loss.alpha
            else:
                if "mse" in self.config.loss:
                    loss_dict['mse_loss'] = (sum(self.mse_loss(sar_content,rgb_content) for sar_content,rgb_content in zip(sar_content[2:],rgb_content[2:])) / 2)*0.1           
            mem = torch.cuda.max_memory_allocated() // 1024 // 1024
            loss_dict['mem'] = torch.from_numpy(np.array([mem], dtype=np.float32)).to(self.device)
            return loss_dict
        if self.infer_rgb == True:
            cls_prob = torch.softmax(rgb_cls_pred, dim=1)
        else:
            cls_prob = torch.softmax(sar_cls_pred, dim=1)
        return cls_prob
    
    def test(self,x):
        content = self.en_content(x)
        if isinstance(content,tuple):
            if 2 in self.config.encoder.wt_layer:
                content,w_list = content[0],content[1]
        specific = self.en_sar(x)
        fuse = self.Project(specific,content)
        fuse_f = self.Concat(fuse,content)
        cls_pred = self.decoder_sar(fuse_f)
        return cls_pred

    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).to(self.device)
        ones = ones.index_select(0, label)
        size.append(N)
        ones = ones.view(*size)
        ones = ones.transpose(2, 3)
        ones = ones.transpose(1, 2)
        return ones

    
    def set_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].set_mask_matrix()

    def reset_mask_matrix(self):
        for index in range(len(self.cov_matrix_layer)):
            self.cov_matrix_layer[index].reset_mask_matrix()

    def conduct_mask_matrix(self):
        val_dataloader = make_dataloader(self.config.data)
        self.en_content.eval()
        self.reset_mask_matrix()
        with torch.no_grad():
            for img,_ in tqdm(val_dataloader):
                img = img.to(self.device)
                sar = img[:,:self.config.sar_in_channels,:,:]
                rgb = img[:,self.config.sar_in_channels:,:,:]
                _,sar_w_arr = self.en_content(sar)
                org_conv=self.en_content.resnet.conv1
                self.en_content.resnet.conv1=self._modules['rgb_conv1']
                _,rgb_w_arr = self.en_content(rgb)
                self.en_content.resnet.conv1=org_conv

                for index, (sar_f_map,rgb_f_map) in enumerate(zip(sar_w_arr,rgb_w_arr)):
                    B, C, H, W = sar_f_map.shape  # i-th feature size (B X C X H X W)
                    HW = H * W
                    sar_f_map = sar_f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
                    rgb_f_map = rgb_f_map.contiguous().view(B, C, -1)
                    eye, reverse_eye = self.cov_matrix_layer[index].get_eye_matrix()
                    for b in range(B):
                        f_map = torch.concat([torch.unsqueeze(sar_f_map[b],dim=0),torch.unsqueeze(rgb_f_map[b],dim=0)],dim=0)
                        f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW - 1) + (self.eps * eye)  # 2 X C X C / HW
                        off_diag_elements = f_cor * reverse_eye
                        self.cov_matrix_layer[index].set_variance_of_covariance(torch.var(off_diag_elements, dim=0))
            self.set_mask_matrix()
        self.en_sar.train()
        self.en_rgb.train()



    def mmr_Loss(self,sar_w_arr,rgb_w_arr):
        wt_loss = torch.FloatTensor([0]).cuda()
        for index, (sar_f_map,rgb_f_map) in enumerate(zip(sar_w_arr,rgb_w_arr)):
            eye, mask_matrix, _, num_sensitive = self.cov_matrix_layer[index].get_mask_matrix()
            sar_loss = instance_whitening_loss(sar_f_map, eye, mask_matrix, num_sensitive)
            rgb_loss = instance_whitening_loss(rgb_f_map, eye, mask_matrix, num_sensitive)
            wt_loss = wt_loss + sar_loss + rgb_loss
        wt_loss = wt_loss / (len(sar_w_arr)*2)
        return wt_loss

    def set_default_config(self):
        self.config.update(dict(
            encoder=dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                load_rgb=0,
                freeze_at=0,
                # 8, 16 or 32
                output_stride=32,
                with_cp=(False, False, False, False),
                stem3_3x3=False,
                norm_layer=nn.BatchNorm2d,
                wt = [0,0,0,0,0,0,0]
            ),
            fpn=dict(
                in_channels_list=(256, 512, 1024, 256),
                out_channels=256,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4,
                norm_fn=nn.BatchNorm2d,
                num_groups_gn=None
            ),
            loss=dict(
            )
        ))