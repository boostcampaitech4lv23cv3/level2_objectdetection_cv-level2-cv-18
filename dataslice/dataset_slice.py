from sahi.slicing import slice_coco
from sahi.utils.file import load_json
from pycocotools.coco import COCO
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Submission macro / USE : python submission [WORK_DIR] [EPOCH_NAME]')
    parser.add_argument('image_dir', help='image_dir (ex: /opt/ml/dataset/)')
    parser.add_argument('coco_annotation_file_path', help='coco_annotation_file_path (ex: /opt/ml/dataset/train_remove_0.json')
    parser.add_argument('output_dir', help='output_dir (ex: /opt/ml/dataset/train_sliced/)')
    parser.add_argument('--image_size', type=tuple, required=False,default=(640, 640), help='image_size')
    args = parser.parse_args()
    return args

def main(args):
    coco_annotation_file_path = args.coco_annotation_file_path
    image_dir = args.image_dir
    output_coco_annotation_file_name = 'slice_train'
    output_dir = args.output_dir
    coco = COCO(coco_annotation_file_path)
    image_num_list = list(coco.getImgIds())
    for i in image_num_list:
        image_info = coco.loadImgs(coco.getImgIds(imgIds=int(i)))[0]
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)
        for a in anns:
            if a['area'] < 1:
                print(image_info)
                print("[error] plz remove 0 area")
                return 0
    _, coco_path = slice_coco(
        coco_annotation_file_path=coco_annotation_file_path,
        image_dir=image_dir,
        output_coco_annotation_file_name=output_coco_annotation_file_name,
        ignore_negative_samples=False,
        output_dir=output_dir,
        slice_height=args.image_size[1],
        slice_width=args.image_size[0],
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        min_area_ratio=0.1,
        verbose=False
    )
    print(coco_path)

if __name__ == "__main__":
    args = parse_args()
    main(args)