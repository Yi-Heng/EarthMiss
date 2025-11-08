import ever as er
from ever.core.builder import make_model, make_dataloader
import torch
import os
import random
from data.EarthMiss import COLOR_MAP
from tqdm import tqdm
from module.viz import VisualizeSegmm
from module.slide_test import slide_inference
import logging
from ever.core.checkpoint import load_model_state_dict_from_ckpt
from ever.core.config import import_config
import numpy as np
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from datetime import datetime



def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser(description='Eval methods')
parser.add_argument('--ckpt_path',  type=str,
                    help='ckpt path', default='./log/deeplabv3p.pth')
parser.add_argument('--config_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--vis_path',  type=str,
                    help='config path', default='baseline.deeplabv3p')
parser.add_argument('--modality',  type=str,
                    help='modality', default='SAR')
parser.add_argument('--tta',  type=bool,
                    help='use tta', default=False)
args = parser.parse_args()

seed_torch(2333)
logger = logging.getLogger(__name__)

logging.getLogger().setLevel(logging.INFO) 


er.registry.register_all()

def evaluate(ckpt_path, config_path='base.hrnetw32', vis_path='model',use_tta=False,modality='SAR',slide_test=False):
    cfg = import_config(config_path)

    model_state_dict = load_model_state_dict_from_ckpt(ckpt_path)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    log_dir = os.path.dirname(ckpt_path)
    
    if modality == 'SAR':
        cfg['model']['params']['modality'] = [1,0]
        cfg.model.params.load_rgb = False
    elif modality == 'RGB':
        cfg['model']['params']['modality'] = [0,1]
        cfg.model.params.load_rgb = True
    else:
        cfg['model']['params']['modality'] = [1,1]
    model = make_model(cfg['model'])
    
    if torch.cuda.is_available():
        local_rank=int(os.environ["LOCAL_RANK"])  
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://" 
        )
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.load_state_dict(model_state_dict)
    model.to(device)
    
    if dist.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)
    
    test_dataloader = make_dataloader(cfg['data']['test'])
    model.eval()
    classes = cfg.model.params.num_classes if cfg.model.params.num_classes != 1 else 2
    metric_op = er.metric.PixelMetric(classes, logdir=log_dir, logger=logger,class_names=list(COLOR_MAP.keys()))
    vis_dir = os.path.join(log_dir,vis_path)
    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.cuda()
            y_true = gt['cls']
            y_true = y_true.cpu()
            if slide_test:
                pred = slide_inference(model, img,gt, slide_test)
            else:
                pred = model(img,gt)
            pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], pred[valid_inds])

            for clsmap, imname in zip(pred, gt['fname']):
                viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
    td = metric_op.summary_all()
    if not dist.is_initialized() or dist.get_rank() == 0:
        with open(os.path.join(log_dir, 'metric_summary.log'), 'a') as log_file:
            log_file.write(f"\n{'='*50}\n")
            log_file.write(f"Evaluation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Checkpoint: {os.path.basename(ckpt_path)}\n")
            log_file.write(str(td))
            log_file.write("\n")
    torch.cuda.empty_cache()




if __name__ == '__main__':
    # ckpt_path = './log/deeplabv3p.pth'
    # config_path = 'baseline.deeplabv3p'
    slide_test = dict(
        crop_size=(512,512),
        stride=(341,341)
    )
    model1 = evaluate(args.ckpt_path, args.config_path, args.vis_path,args.tta,args.modality,slide_test=slide_test)
