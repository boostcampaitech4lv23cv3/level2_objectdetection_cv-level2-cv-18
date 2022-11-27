from mmcv import Config
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
from sahi.predict import get_prediction, get_sliced_prediction, predict
import pandas as pd
from pandas import DataFrame
from pycocotools.coco import COCO
import numpy as np
import argparse
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser(description='Submission macro / USE : python submission [WORK_DIR] [EPOCH_NAME]')
    parser.add_argument('work_dir', help='work_dir')
    parser.add_argument('model_name', help='checkpoint file name in work_dir')
    parser.add_argument('test_data_path', help='test_data_path')
    parser.add_argument('--image_size', type=tuple, required=False,default=(640, 640), help='image_size')
    args = parser.parse_args()
    return args

def main(args) :
    test_data_path= args.test_data_path
    config_name = [path for path in os.listdir(args.work_dir) if path.endswith('.py')][0]
    model_path = os.path.join(args.work_dir, args.model_name)
    config_path = os.path.join(args.work_dir, config_name)
    print('Selected config_path :', config_path)
    print('Selected model_path :', model_path)
    print('Selected test_data_path :', test_data_path)

    model_type = "mmdet"
    model_path = model_path
    model_config_path = config_path
    model_device = 'cuda:0'
    model_confidence_threshold = 0.5
    slice_height = args.image_size[0]
    slice_width = args.image_size[1]
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2
    source_image_dir = test_data_path
    project = args.work_dir
    name = 'exp'
    novisual = True
    
    predict(
        model_type=model_type,
        model_path=model_path,
        model_config_path=config_path,
        model_device=model_device,
        model_confidence_threshold=model_confidence_threshold,
        source=source_image_dir,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        export_pickle=True,
        project=project,
        name=name,
        novisual=novisual
    )

    cfg = Config.fromfile(config_path)
    class_num = 10
    prediction_strings = []
    file_names = []
    coco = COCO(cfg.data.test.ann_file)
    pickle_list = os.listdir(f"{project}/{name}3/pickles")
    prediction_strings = []
    file_names = []
    for file in sorted(pickle_list):
        img_num, _ = file.split(".")
        image_info = coco.loadImgs(coco.getImgIds(imgIds=int(img_num)))[0]
        prediction_string = ''
        with open(f"{project}/{name}3/pickles/{file}","rb") as fr:

            data = pickle.load(fr)
            for d in data:
                coco_prediction = d.to_coco_prediction()
                coco_prediction_json = coco_prediction.json
                prediction_string += str(coco_prediction_json['category_id']) + ' ' + str(coco_prediction_json['score']) + ' ' + str(float(coco_prediction_json['bbox'][0])) + ' ' + str(float(coco_prediction_json['bbox'][0])+float(coco_prediction_json['bbox'][2])) + ' ' + str(float(coco_prediction_json['bbox'][1])) + ' ' + str(float(coco_prediction_json['bbox'][1])+float(coco_prediction_json['bbox'][3])) + ' '

        prediction_strings.append(prediction_string)
        file_names.append(image_info['file_name'])
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv(os.path.join(args.work_dir, f'{args.model_name}_submission_slice.csv'), index=None)  # type: ignore
    print("done")
    os.system(f"rm -rf {project}/{name}")

    

if __name__ == "__main__":
    args = parse_args()
    main(args)