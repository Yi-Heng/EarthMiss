import ever as er
import torch
import numpy as np
import os
from data.EarthMiss import COLOR_MAP
from tqdm import tqdm
import random
from module.slide_test import slide_inference
from module.viz import VisualizeSegmm
from module.baseline.MetaRS import MetaRS
from ever import registry


def evaluate_cls_fn(self, test_dataloader, config=None):
    self.model.eval()
    classes = self.model.module.config.num_classes if self.model.module.config.num_classes != 1 else 2
    metric_op = er.metric.PixelMetric(classes, logdir=self._model_dir, logger=self.logger,class_names=list(COLOR_MAP.keys()))

    vis_dir = os.path.join(self._model_dir, 'vis-{}'.format(self.checkpoint.global_step))

    palette = np.array(list(COLOR_MAP.values())).reshape(-1).tolist()
    viz_op = VisualizeSegmm(vis_dir, palette)

    with torch.no_grad():
        for img, gt in tqdm(test_dataloader):
            img = img.to(torch.device('cuda'))
            y_true = gt['cls']
            y_true = y_true.cpu()
            if self.config.get('slide_test', False):
                pred = slide_inference(self.model,img,gt, config['slide_test'])
            else:
                pred = self.model(img,gt)

            y_pred = pred.argmax(dim=1).cpu()

            valid_inds = y_true != -1
            metric_op.forward(y_true[valid_inds], y_pred[valid_inds])

            for clsmap, imname in zip(y_pred, gt['fname']):
                if "tif" in imname:
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('tif', 'png'))
                elif "jpg" in imname:
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname.replace('jpg', 'png'))
                else:
                    viz_op(clsmap.cpu().numpy().astype(np.uint8), imname)
    metric_op.summary_all()
    torch.cuda.empty_cache()



def register_evaluate_fn(launcher):
    launcher.override_evaluate(evaluate_cls_fn)


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    seed_torch(2333)
    trainer = er.trainer.get_trainer()()
    trainer.run(after_construct_launcher_callbacks=[register_evaluate_fn])
