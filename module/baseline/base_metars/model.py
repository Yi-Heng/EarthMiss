from torch import nn
import torch
from segmentation_models_pytorch.base import modules as md
import numpy as np





class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
class FeatureFusionModuleSCSE_V1(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan * 2, out_chan, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)
                
class FeatureFusionModuleSCSE_V2(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()

        self.scse_1 = md.SCSEModule(in_chan)
        self.scse_2 = md.SCSEModule(in_chan)
        self.convblk = ConvBNReLU(in_chan * 2, out_chan, ks=1, stride=1, padding=0)
        self.scse = md.SCSEModule(out_chan)
        self.init_weight()

    def forward(self, fsp, fcp):
        fsp = self.scse_1(fsp)
        fcp = self.scse_2(fcp)
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        feat_out = self.scse(feat)
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class Fusespfnv1(nn.Module):
    def __init__(self, inchan_list):
        super(Fusespfnv1, self).__init__()
        fuse_module_list = []
        for in_chan in inchan_list:
            fuse_module_list.append(FeatureFusionModuleSCSE_V1(in_chan,in_chan))
        self.fuse_module_list=nn.ModuleList(fuse_module_list)

    def forward(self, unimodal_list,share_list):
        fuse_fpn_list=[]
        for i in range(len(unimodal_list)):
            out = self.fuse_module_list[i](unimodal_list[i],share_list[i])
            fuse_fpn_list.append(out)
        return fuse_fpn_list

class Fusespfnv2(nn.Module):
    def __init__(self, inchan_list):
        super(Fusespfnv2, self).__init__()
        fuse_module_list = []
        for i,in_chan in enumerate(inchan_list):
            in_chan = in_chan
            fuse_module_list.append(FeatureFusionModuleSCSE_V2(in_chan,in_chan))
        self.fuse_module_list=nn.ModuleList(fuse_module_list)
    def forward(self, unimodal_list,share_list):
        fuse_fpn_list=[]
        for i in range(len(unimodal_list)):
            out = self.fuse_module_list[i](unimodal_list[i],share_list[i])
            fuse_fpn_list.append(out)
        return fuse_fpn_list
class Fusespfnv2_res(nn.Module):
    def __init__(self, inchan_list):
        super(Fusespfnv2_res, self).__init__()
        fuse_module_list = []
        for i,in_chan in enumerate(inchan_list):
            in_chan = in_chan
            fuse_module_list.append(FeatureFusionModuleSCSE_V2(in_chan,in_chan))
        self.fuse_module_list=nn.ModuleList(fuse_module_list)
    def forward(self, specific_list,share_list):
        fuse_fpn_list=[]
        for i in range(len(specific_list)):
            out = self.fuse_module_list[i](specific_list[i],share_list[i])
            out += specific_list[i]
            fuse_fpn_list.append(out)
        return fuse_fpn_list

class Clssify(nn.Module):
    def __init__(self, in_c):
        super(Clssify,self).__init__()
        self.classfily = nn.Sequential(
            nn.Conv2d( in_c,  in_c//4, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d( in_c//4 , 2, kernel_size=3, padding=1),
        
        )
    def forward(self,x):
        return self.classfily(x)

class dco_Clssify(nn.Module):
    def __init__(self, in_c):
        super(dco_Clssify,self).__init__()
        self.classfily = nn.Sequential(
            nn.Conv2d(in_c,  in_c//4, 1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d( in_c//4,2, 1),
        )
    def forward(self,x):
        return self.classfily(x)     

class Feature_clssify(nn.Module):
    def __init__(self, in_channel_list):
        super(Feature_clssify,self).__init__()
        FC=[]
        for inc in in_channel_list:
            FC.append(Clssify(inc))
        self.fc=nn.ModuleList(FC)


    def forward(self,x):
        outs=[]
        for i in range(len(x)):
            outs.append(self.fc[i](x[i]))
        return outs

class dco_Feature_clssify(nn.Module):
    def __init__(self, in_channel_list):
        super(dco_Feature_clssify,self).__init__()
        FC=[]
        for inc in in_channel_list:
            FC.append(dco_Clssify(inc))
        self.fc=nn.ModuleList(FC)


    def forward(self,x):
        outs=[]
        for i in range(len(x)):
            outs.append(self.fc[i](x[i]))
        return outs

class RandomDropModality(nn.Module):

    def __init__(self,modality_num,total_iterations):
        super(RandomDropModality, self).__init__()
        self.modality_num = modality_num
        self.prob = np.array([1/(self.modality_num+1)]*(self.modality_num+1))
        self.total_iterations = total_iterations
    def set_prob(self,iterations,power=0.9):
        p = (1-iterations/self.total_iterations)**power
        complete_combination_p = max(p,1/(1+self.modality_num))
        other_combination_p = (1-complete_combination_p)/self.modality_num
        prob = [complete_combination_p]+[other_combination_p]*self.modality_num
        self.prob = np.array(prob)
    def forward(self,combination,*modalitys):
        #tye of modalitys is tuple
        modalitys = list(modalitys)
        modality_num = len(modalitys)
        if combination != []:
            return combination
        else:
            combinations = []
            modaliy_combinations = [[1]*len(modalitys) for _ in range(modality_num+1)]
            for i in range(1,modality_num+1):
                modaliy_combinations[i][i-1] = 0
            index_list = [i for i in range(self.modality_num+1)]
            index = np.random.choice(index_list, size=1, replace=True, p=self.prob)[0]
            combinations = modaliy_combinations[index]
        return combinations


class Concat(nn.Module):
    def __init__(self):
        super(Concat, self).__init__()
    def forward(self,sar_fe_list,rgb_fe_list):
        fuse_fe_list = []
        for i in range(len(sar_fe_list)):
            fuse_fe_list.append(torch.concat((sar_fe_list[i],rgb_fe_list[i]),dim=1))
        return fuse_fe_list