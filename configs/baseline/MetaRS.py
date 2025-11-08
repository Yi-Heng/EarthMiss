from module.baseline.base import default_conv_block

from configs.base.EarthMiss import train, test, data, optimizer, learning_rate

config = dict(
    model=dict(
        type='MetaRS',
        params=dict(
            sar_in_channels=1,rgb_in_channels=3,
            infer_rgb = False,
            in_channels_list=(0,0,64,256,512,1024,2048),
            fuse_inchan_list = (256,512,1024,2048),
            relax_denom = 0,
            clusters = 10,
            begin_mmr_iter = 1600,
            encoder = dict(
                resnet_type='resnet50',
                include_conv5=True,
                batchnorm_trainable=True,
                pretrained=True,
                freeze_at=0,
                output_stride=32,
                wt_layer = [0,0,2,2,2,0,0],
                num_classes = 1
            ),
            fpn=dict(
                in_channels_list=(512,1024,2048,4096),
                out_channels=256,
                conv_block=default_conv_block,
                top_blocks=None,
            ),
            decoder=dict(
                in_channels=256,
                out_channels=128,
                in_feat_output_strides=(4, 8, 16, 32),
                out_feat_output_stride=4
            ),
            scale_factor = 16,
            num_classes = 8,
            loss=dict(
                ce=dict(),
                ignore_index=-1,
                mmr = dict(),
                mse = dict(),
                alpha = 1.25
            ),
            data = data["val"]
        )
    ),
    data=data,
    optimizer=optimizer,
    learning_rate=learning_rate,
    train=train,
    test=test
)
