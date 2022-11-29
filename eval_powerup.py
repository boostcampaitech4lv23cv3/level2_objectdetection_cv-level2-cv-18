import mmcv
from mmcv import Config
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmdet.utils import get_device
import os
from mmcv.parallel import MMDataParallel
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse

def main(args):
    root ='/opt/ml/dataset'
    json_dir = os.path.join(root, 'train.json')
    work_dir = args.work_dir
    epoch_name = args.epoch_name
    config_name = [path for path in os.listdir(work_dir) if path.endswith('.py')][0]
    path_epoch = os.path.join(work_dir, epoch_name)
    path_config = os.path.join(work_dir, config_name)
    print('Selected Config :', path_config)
    print('Selected Epoch :', path_epoch)

    # init config
    cfg = Config.fromfile(path_config)
    cfg.work_dir = work_dir
    cfg.data.samples_per_gpu = 4
    cfg.gpu_ids = [1]
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None

    # build dataset & dataloader
    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
            dataset,
            samples_per_gpu=cfg.data.samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

    # load model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg')) # build detector
    #device = device = get_device()
    _ = load_checkpoint(model, path_epoch, map_location='cpu') # ckpt load
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])

    # inference
    output = single_gpu_test(model, data_loader, show_score_thr=0.05) # output 계산

    # save
    class_num = 10
    prediction_strings = []
    file_names = []
    train_ids = []
    coco = COCO(cfg.data.val.ann_file)

    for i, out in enumerate(output):
        image_info = coco.dataset['images'][i]
        file_names.append(image_info['file_name'])
        train_ids.append(image_info['id'])

        prediction_string = ''
        for j in range(class_num):
            for o in out[j]:
                prediction_string = ' '.join([prediction_string, str(j), str(o[4]), str(o[0]), str(o[1]), str(o[2]), str(o[3])])
        prediction_strings.append(prediction_string)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission['train_id'] = train_ids
    submission.to_csv(os.path.join(cfg.work_dir, f'{epoch_name}_eval.csv'), index=None)  # type: ignore
    print(submission.head())

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--epoch_name', type=str, default='latest.pth')
    args = parser.parse_args()

    main(args)