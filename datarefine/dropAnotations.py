import os
import json
import argparse
import pandas
import copy
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-dr', help='dataset folder path', type=str, default="/opt/ml/dataset")
    parser.add_argument('source_json', 
                        type=str,
                        default='train.json',
                        help='target json file name in dataset root folder')
    parser.add_argument('-rr', help='result folder path', type=str, default="/opt/ml/dataset")
    parser.add_argument('result_json',
                        type=str,
                        default='train.json',
                        help='result json file to save in result root folder')
    
    args = parser.parse_args()
    return args


def main(args:argparse.Namespace):
    source_path = os.path.join(args.dr, args.source_json)
    result_path = os.path.join(args.rr, args.result_json)
    with open(source_path) as js:
        source_json = json.load(js)
        result_json = copy.deepcopy(source_json)
        refined_annotations = []
        annotations = source_json['annotations']
        for i, ann in tqdm(enumerate(annotations)):
            if ann['area'] < 100 :
                print(' - droped :', ann['id'], ' cz small area')
                continue;
            elif ann['bbox'][2] < 9:
                print(' - droped :', ann['id'], ' cz small width')
                continue;
            elif ann['bbox'][3] < 9:
                print(' - droped :', ann['id'], ' cz small height')
                continue;
            refined_annotations.append(ann)
        result_json['annotations'] = refined_annotations
        with open(result_path, "w") as json_file:
            json.dump(result_json, json_file)
            print(' * new json dataset has been created at ', result_path)
    # source = json.load(source_path)

    return 1

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)