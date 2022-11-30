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
    parser.add_argument('-dmn', help='min drop', type=bool, default=True)
    parser.add_argument('-dmx', help='max drop', type=bool, default=True)
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
            if args.dmn == True:
                if ann['area'] < 100 :
                    print(' - droped :', ann['id'], ' cz small area')
                    continue;
                elif ann['bbox'][2] < 20 or ann['bbox'][3] < 20:
                    print(' - droped :', ann['id'], ' cz small width')
                    continue;
            if args.dmx == True:
                if ann['bbox'][2] > 1000 or ann['bbox'][3] > 1000: 
                    if ann['area'] > 1000*900 :
                        # 0 : general / 1 : paper / 7 : plastic bag
                        if ann['category_id'] == 0 or ann['category_id'] == 1 or ann['category_id'] == 7:
                            print(' - droped :', ann['id'], ' cz big WH and big area')
                            continue;
            
            refined_annotations.append(ann)
        print(' * original size : ', len(annotations), ' / result size : ', len(refined_annotations), ' / total removed : ', len(annotations) - len(refined_annotations))
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