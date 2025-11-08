from albumentations import Compose, OneOf, Normalize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop,RandomBrightnessContrast
import ever as er
from ..metadata.EarthMiss import*

data = dict(
    train=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            sensors=('SAR','RGB'),
            transforms=Compose([
                RandomCrop(512, 512),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Normalize(mean=sar_mean_v+rgb_mean_v,
                          std=sar_std_v+rgb_std_v,
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()
            ]),
            CV=dict(k=10, i=-1),
            distributed=True,
            training=True,
            batch_size=8,
            num_workers=4,
        ),
    ),
    val=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            sensors=('SAR','RGB'),
            transforms=Compose([
                Normalize(mean=sar_mean_v+rgb_mean_v,
                          std=sar_std_v+rgb_std_v,
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()
            ]),
            CV=dict(k=10, i=-1),
            training=False,
            batch_size=4,
            num_workers=4,
        ),
    ),
    test=dict(
        type='EarthM3DALoader',
        params=dict(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            sensors=('SAR','RGB'),
            transforms=Compose([
                Normalize(mean=sar_mean_v+rgb_mean_v,
                          std=sar_std_v+rgb_std_v,
                          max_pixel_value=1, always_apply=True),
                er.preprocess.albu.ToTensor()
            ]),
            CV=dict(k=10, i=-1),
            distributed=True,
            training=False,
            batch_size=16,
            num_workers=4,
        ),
    ),
)

optimizer = dict(
    type='adamw',
    params=dict(
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
)


learning_rate = dict(
    type='poly',
    params=dict(
        base_lr=1e-4,
        power=0.9,
        max_iters=15000,
    )
)
train = dict(
    forward_times=1,
    num_iters=15000,
    eval_per_epoch=True,
    summary_grads=False,
    summary_weights=False,
    distributed=True,
    apex_sync_bn=True,
    sync_bn=True,
    eval_after_train=True,
    log_interval_step=50,
    save_ckpt_interval_epoch=20,
    eval_interval_epoch=20,
    distributed_evaluate=True,
)

test = dict(

)

