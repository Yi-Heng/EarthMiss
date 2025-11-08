import warnings

warnings.filterwarnings('ignore')
from torch.utils.data import Dataset
import glob
import os
from skimage.io import imread
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize, RandomCrop, RandomScale
from albumentations import OneOf, Compose
import ever as er
from collections import OrderedDict
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler
from ever.data import distributed, CrossValSamplerGenerator
import numpy as np
import logging
from PIL import Image
import tifffile
from skimage.io import imread
import cv2
import torch
import random
logger = logging.getLogger(__name__)

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),#白色
    Building=(255, 0, 0),#红色
    Road=(255, 255, 0),#黄色
    Water=(0, 0, 255),#蓝色
    Barren=(159, 129, 183),#紫色
    Forest=(0, 255, 0),#绿色
    Agricultural=(255, 195, 128),#深一点的黄色
    Playground=(165,0,165)#深紫色
)

LABEL_MAP = OrderedDict(
    IGNORE=-1,
    Background=0,
    Building=1,
    Road=2,
    Water=3,
    Barren=4,
    Forest=5,
    Agricultural=6,
    Playground=7
)


def reclassify(cls):
    new_cls = np.ones_like(cls, dtype=np.int8) * -1
    for idx, label in enumerate(LABEL_MAP.values()):
        new_cls = np.where(cls == idx, np.ones_like(cls) * label, new_cls)
    return new_cls

def if_same_image(file1,file2,prex1,prex2):
    file1_id = os.path.basename(file1)
    file2_id = os.path.basename(file2)
    file1_id = file1_id.replace(prex1, prex2)   
    return file1_id == file2_id

class EarthM3(Dataset):
    def __init__(self, image_dir,mask_dir,transforms=None,sensors=('','')):
        self.image_filepath_list = []
        self.cls_filepath_list = []
        self.sensor = sensors

        if isinstance(image_dir, list) and isinstance(mask_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)
        elif isinstance(image_dir, list) and not isinstance(mask_dir, list):
            for img_dir_path in image_dir:
                self.batch_generate(img_dir_path, mask_dir)
        else:
            self.batch_generate(image_dir, mask_dir)
        self.transforms=transforms
        logger.info('%s -- Dataset images: %d' % (os.path.dirname(image_dir[0]), len(self.image_filepath_list)))
        logger.info('%s -- Dataset masks: %d' % (os.path.dirname(image_dir[0]), len(self.cls_filepath_list)))
        self.image_filepath_list.sort()
        self.cls_filepath_list.sort()
        self.downsampling_rates = [2,4,8,16,32]


    def batch_generate(self, image_dir, mask_dir):
        sar_image_dir=os.path.join(image_dir,self.sensor[0])
        rgb_image_dir=os.path.join(image_dir,self.sensor[1])
        sar_image_filepath_list = glob.glob(os.path.join(sar_image_dir, '*.tif'))
        rgb_image_filepath_list = glob.glob(os.path.join(rgb_image_dir, '*.tif'))
        sar_image_filepath_list += glob.glob(os.path.join(sar_image_dir, '*.png'))
        rgb_image_filepath_list += glob.glob(os.path.join(rgb_image_dir, '*.png'))
        cls_filepath_list=[]

        if mask_dir is not None:
            cls_filepath_list += glob.glob(os.path.join(mask_dir, '*.tif'))
            cls_filepath_list += glob.glob(os.path.join(mask_dir, '*.png'))

        self.cls_filepath_list += cls_filepath_list
        sar_image_filepath_list.sort()
        rgb_image_filepath_list.sort()
        cls_filepath_list.sort()

        assert len(sar_image_filepath_list) == len(rgb_image_filepath_list),"{0}'s nums!={1}'nums".format(self.sensor[0],self.sensor[1])
        assert len(sar_image_filepath_list) == len(cls_filepath_list), "{0}'s nums!=cls'nums".format(self.sensor[0])
        for i in range(len(sar_image_filepath_list)):
            assert if_same_image(sar_image_filepath_list[i],rgb_image_filepath_list[i],"_SAR","")
            assert if_same_image(cls_filepath_list[i], rgb_image_filepath_list[i], "_mask", ""), \
            f"mask is {os.path.basename(cls_filepath_list[i])} , and rgb is {os.path.basename(rgb_image_filepath_list[i])}"
            self.image_filepath_list+=[(sar_image_filepath_list[i],rgb_image_filepath_list[i])]

    


    def get_prompt(self,mask,num_points = 1):
        max_labels = 8
        point_labels = []
        point_coords = []
        for i in range(max_labels):
            indices = np.argwhere(mask == i)
            input_point = []
            input_label = []
            if indices.size(1) < num_points:
                for j in range(indices.size(1)):
                    coord = [indices[0][j],indices[1][j]]
                    input_point.append(coord)
                    input_label.append(1)
                while len(input_point) < num_points:
                    if indices.size(1) != 0:
                        input_point.append(coord[:])
                    else:
                        input_point.append([256,256])
                    input_label.append(1)
            else:
                index = random.sample(range(0, len(indices[0])) , num_points)                           
                for i in index:
                    coord = [indices[0][i], indices[1][i]]
                    input_point.append(coord[:])
                    input_label.append(1)
            point_coords.extend(input_point[:])
            point_labels.extend(input_label[:])
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.float)
        return coords_torch, labels_torch

    def __getitem__(self, idx):
        if len(self.cls_filepath_list) > 0:
            assert if_same_image(self.cls_filepath_list[idx],self.image_filepath_list[idx][1],"_mask",""), \
            f"mask is {os.path.basename(self.cls_filepath_list[idx])} , and rgb is {os.path.basename(self.image_filepath_list[idx][1])}" \
            f"SAR is {os.path.basename(self.image_filepath_list[idx][0])} and RGB is {os.path.basename(self.image_filepath_list[idx][1])}"

        ms_images = []
        for fp in self.image_filepath_list[idx]:
            image_arr = imread(fp)
            if len(image_arr.shape) == 2:
                image_arr = image_arr[:,:,np.newaxis]
            ms_images.append(image_arr)
        if len(ms_images) > 1:
            ms_images = np.concatenate(ms_images, axis=2)
        else:
            ms_images = ms_images[0]
        
        if len(self.cls_filepath_list) > 0:
            mask_dir = self.cls_filepath_list[idx]
            mask = imread(mask_dir).astype(np.int8)
            mask = mask-1
            if self.transforms is not None:
                blob = self.transforms(image=ms_images, mask=mask)
                ms_images = blob['image']
                mask = blob['mask']


            points = self.get_prompt(mask) 

            h, w = mask.shape
            sample = dict()

            for rate in self.downsampling_rates:
                label_down = cv2.resize(mask.numpy(), (w // rate, h // rate),
                                        interpolation=cv2.INTER_NEAREST)
                sample[rate] =  torch.from_numpy(label_down)
            return ms_images, dict(cls=mask, mask_down = sample,mask_id=os.path.basename(self.cls_filepath_list[idx]),points=points,fname=os.path.basename(self.image_filepath_list[idx][1]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=ms_images)
                ms_images = blob['image']

        return ms_images, dict(fname=os.path.basename(self.image_filepath_list[idx][0]))
    def __len__(self):
        return len(self.image_filepath_list)


@er.registry.DATALOADER.register()
class EarthM3DALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config):
        ConfigurableMixin.__init__(self, config)
        dataset = EarthM3(self.config.image_dir, self.config.mask_dir, self.config.transforms,self.config.sensors)
        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = distributed.StepDistributedSampler(dataset) if self.config.training else distributed.DistributedNonOverlapSeqSampler(dataset)
            #sampler = distributed.StepDistributedSampler(dataset) if self.config.training else SequentialSampler(dataset)
        super(EarthM3DALoader, self).__init__(dataset,
                                           self.config.batch_size,
                                           sampler=sampler,
                                           num_workers=self.config.num_workers,
                                           pin_memory=True,
                                           drop_last=False
                                           )

    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
        ))
