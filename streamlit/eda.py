import streamlit as st

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import cv2

from pycocotools.coco import COCO
import mmdet.core.visualization.image as bbox_util

# Configuration
gt_path = '/opt/ml/dataset/train.json'
train_image_path = '/opt/ml/dataset'
classes = ["General trash", "Paper", "Paper pack", "Metal", "Glass",
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]


# Session Manager
def init_session_manager():
    if "complete_pred" not in st.session_state:
        st.session_state.complete_pred = False

    if not st.session_state.complete_pred:
        st.session_state.confusion_matrix_image = None

    if "train_raw_dataset" not in st.session_state:
        st.session_state.train_raw_dataset = COCO(gt_path)

    if "train_dataframe" not in st.session_state:
        st.session_state.train_dataframe = None

    if "train_bboxed_image" not in st.session_state:
        st.session_state.train_bboxed_image = None

    if "work_dirs" not in st.session_state:
        st.session_state.work_dirs = "./work_dirs/mask-rcnn"

    if "pth_file" not in st.session_state:
        st.session_state.pth_file = "best_bbox_mAP_epoch_300.pth"

    if "val_file" not in st.session_state:
        st.session_state.val_file = "best_bbox_mAP_epoch_300.pth_submission.csv"

    if "selected_train_image_id" not in st.session_state:
        st.session_state.selected_train_image_id = 0

    if "train_bbox_size" not in st.session_state:
        st.session_state.train_bbox_size = ['Small', 'Medium', 'Large']

    if "train_bbox_class" not in st.session_state:
        st.session_state.train_bbox_class = classes


# Sub functions
def box_iou_calc(boxes1, boxes2):
    # <https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py>
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


class ConfusionMatrix:
    def __init__(self, num_classes: int, conf_threshold: float = 0.3, iou_threshold: float = 0.5):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = conf_threshold
        self.IOU_THRESHOLD = iou_threshold

    def plot(self, file_name='./figure.jpg', names=None):
        if names is None:
            names = ['General trash', "Paper", "Paper pack", "Metal", "Glass",
                     "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing"]

        try:
            array = self.matrix / (self.matrix.sum(0).reshape(1, self.num_classes + 1) + 1E-6)  # normalize
            array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

            fig = plt.figure(figsize=(12, 9), tight_layout=True)
            sn.set(font_scale=1.0 if self.num_classes < 50 else 0.8)  # for label size
            labels = (0 < len(names) < 99) and len(names) == self.num_classes  # apply names to ticklabels
            sn.heatmap(array, annot=self.num_classes < 30, annot_kws={"size": 8}, cmap='Blues', fmt='.2f', square=True,
                       xticklabels=names + ['background FP'] if labels else "auto",
                       yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
            fig.axes[0].set_xlabel('True')
            fig.axes[0].set_ylabel('Predicted')
            fig.savefig(file_name, dpi=250)
        except Exception as e:
            print(e)
            pass

    def process_batch(self, detections, labels: np.ndarray):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        gt_classes = labels[:, 0].astype(np.int16)

        try:
            detections = detections[detections[:, 4] > self.CONF_THRESHOLD]
        except IndexError or TypeError:
            # detections are empty, end of process
            for i, label in enumerate(labels):
                gt_class = gt_classes[i]
                self.matrix[self.num_classes, gt_class] += 1
            return

        detection_classes = detections[:, 5].astype(np.int16)

        all_ious = box_iou_calc(labels[:, 1:], detections[:, :4])
        want_idx = np.where(all_ious > self.IOU_THRESHOLD)

        all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                       for i in range(want_idx[0].shape[0])]

        all_matches = np.array(all_matches)
        if all_matches.shape[0] > 0:  # if there is match
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(labels):
            gt_class = gt_classes[i]
            if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                detection_class = detection_classes[int(all_matches[all_matches[:, 0] == i, 1][0])]
                self.matrix[detection_class, gt_class] += 1
            else:
                self.matrix[self.num_classes, gt_class] += 1

        for i, detection in enumerate(detections):
            if not all_matches.shape[0] or (all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0):
                detection_class = detection_classes[i]
                self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))


def draw_confusion_matrix():
    conf_mat = ConfusionMatrix(num_classes=10, conf_threshold=0.01, iou_threshold=0.5)
    pred_path = os.path.join(st.session_state.work_dirs, st.session_state.val_file)
    with open(gt_path, 'r') as outfile:
        test_anno = (json.load(outfile))
    pred_df = pd.read_csv(pred_path)
    new_pred = []
    gt = []

    file_names = pred_df['image_id'].values.tolist()  # type: ignore
    bboxes = pred_df['PredictionString'].values.tolist()  # type: ignore
    for i, bbox in enumerate(bboxes):
        if isinstance(bbox, float):
            # print(f'{file_names[i]} empty box')
            pass

    for file_name, bbox in zip(file_names, bboxes):
        new_pred.append([])
        boxes = np.array(str(bbox).split(' '))

        if len(boxes) % 6 == 1:
            boxes = boxes[:-1].reshape(-1, 6)
        elif len(boxes) % 6 == 0:
            boxes = boxes.reshape(-1, 6)
        else:
            raise Exception('error', 'invalid box count')

        for box in boxes:
            new_pred[-1].append([float(box[2]), float(box[3]), float(box[4]), float(box[5]),
                                 float(box[1]), float(box[0])])

    coco = COCO(gt_path)

    for image_id in coco.getImgIds():
        gt.append([])
        image_info = coco.loadImgs(image_id)[0]
        ann_ids = coco.getAnnIds(imgIds=image_info['id'])
        anns = coco.loadAnns(ann_ids)

        file_name = image_info['file_name']

        for ann in anns:
            gt[-1].append([
                float(ann['category_id']),
                float(ann['bbox'][0]),
                float(ann['bbox'][1]),
                float(ann['bbox'][0]) + float(ann['bbox'][2]),
                (float(ann['bbox'][1]) + float(ann['bbox'][3])),
            ])
    for p, g in zip(new_pred, gt):
        conf_mat.process_batch(np.array(p), np.array(g))
    conf_mat.plot(os.path.join(st.session_state.work_dirs, st.session_state.val_file + '.png'))
    st.session_state.confusion_matrix_image = os.path.join(st.session_state.work_dirs,
                                                           st.session_state.val_file + '.png')

    st.session_state.complete_pred = True


def reload_train_data():
    temporary_dict = {'filename': [],
                      'ann_count': [],
                      'ann_min_size': [],
                      'ann_max_size': [],
                      'class_count': [],
                      }

    for train_row in st.session_state.train_raw_dataset.dataset['images']:
        ann_ids = st.session_state.train_raw_dataset.getAnnIds(imgIds=train_row['id'])
        anns = st.session_state.train_raw_dataset.loadAnns(ann_ids)

        filter_class = [classes.index(c) for c in st.session_state.train_bbox_class]
        anns = [a for a in anns if a['category_id'] in filter_class]
        if "Remove ↓1**2" in st.session_state.train_bbox_size:
            anns = [a for a in anns if a['area'] >= 1 ** 2]
        if "Remove ↑1023**2" in st.session_state.train_bbox_size:
            anns = [a for a in anns if a['area'] < 1023 ** 2]
        if "Small" not in st.session_state.train_bbox_size:
            anns = [a for a in anns if a['area'] >= 32 ** 2]
        if "Large" not in st.session_state.train_bbox_size:
            anns = [a for a in anns if a['area'] < 96 ** 2]
        if "Medium" not in st.session_state.train_bbox_size:
            anns = [a for a in anns if (a['area'] < 32 ** 2 or a['area'] >= 96 ** 2)]

        anns = sorted(anns, key=lambda d: d['area'])

        if anns:
            temporary_dict['filename'].append([train_row['file_name']])
            temporary_dict['ann_count'].append(len(anns))
            temporary_dict['ann_min_size'].append(anns[0]['area'])
            temporary_dict['ann_max_size'].append(anns[-1]['area'])
            temporary_dict['class_count'].append(len(set([a['category_id'] for a in anns])))

    st.session_state.train_dataframe = pd.DataFrame.from_dict(temporary_dict)


def draw_train_image():
    image_info = st.session_state.train_raw_dataset.loadImgs(st.session_state.selected_train_image_id)[0]
    ann_ids = st.session_state.train_raw_dataset.getAnnIds(imgIds=image_info['id'])
    anns = st.session_state.train_raw_dataset.loadAnns(ann_ids)

    #st.write(f'{st.session_state.train_bbox_size}')
    #st.write(f'{st.session_state.train_bbox_class}')
    #st.write(f'{anns}')

    filter_class = [classes.index(c) for c in st.session_state.train_bbox_class]
    anns = [a for a in anns if a['category_id'] in filter_class]

    if "Remove ↓1**2" in st.session_state.train_bbox_size:
        anns = [a for a in anns if a['area'] >= 1 ** 2]
    if "Remove ↑1023**2" in st.session_state.train_bbox_size:
        anns = [a for a in anns if a['area'] < 1023 ** 2]
    if "Small" not in st.session_state.train_bbox_size:
        anns = [a for a in anns if a['area'] >= 32**2]
    if "Large" not in st.session_state.train_bbox_size:
        anns = [a for a in anns if a['area'] < 96**2]
    if "Medium" not in st.session_state.train_bbox_size:
        anns = [a for a in anns if (a['area'] < 32**2 or a['area'] >= 96**2)]

    st.session_state.train_anns = anns

    img_path = os.path.join(train_image_path, image_info['file_name'])
    if anns is not None and len(anns) > 0:
        bboxes = np.array([[ann["bbox"][0], ann["bbox"][1],
                            ann["bbox"][0] + ann["bbox"][2], ann["bbox"][1] + ann["bbox"][3]] for ann in anns])
        labels = np.array([ann["category_id"] for ann in anns])

        st.session_state.train_bboxed_image = cv2.cvtColor(bbox_util.imshow_det_bboxes(img_path, bboxes, labels,
                                                                                       bbox_color='voc',
                                                                                       class_names=classes, font_size=12),
                                                           cv2.COLOR_BGR2RGB)
    else:
        st.session_state.train_bboxed_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def update_numin():
    if st.session_state.number_input_image_id < 0:
        st.session_state.number_input_image_id = 0
    elif st.session_state.number_input_image_id > len(st.session_state.train_raw_dataset.dataset['images']) - 1:
        st.session_state.number_input_image_id = len(st.session_state.train_raw_dataset.dataset['images']) - 1
    st.session_state.selected_train_image_id = st.session_state.number_input_image_id
    draw_train_image()


def update_slider():
    st.session_state.selected_train_image_id = st.session_state.slider_image_id
    draw_train_image()


# Main page
st.set_page_config(layout="wide")
tab1, tab2 = st.tabs(["Train Image Viewer", "Valid Image / Confusion Matrix"])

with tab1:
    st.title('Train Image Viewer')
    st.text(f'Current Working Path: {os.getcwd()}    Datasets Path: {gt_path}[{os.path.exists(gt_path)}]')
    if os.path.exists(gt_path):
        init_session_manager()
    else:
        st.error('Not found: Datasets', icon="🚨")
        st.stop()

    # st.write(st.session_state.train_raw_dataset.dataset)

    train_bbox_size = st.multiselect(
        "Select Train Target Size (BBox)",
        ['Remove ↓1**2', 'Small', 'Medium', 'Large', 'Remove ↑1023**2'],
        ['Small', 'Medium', 'Large'],
        on_change=reload_train_data,
        key='train_bbox_size'
    )
    train_bbox_class = st.multiselect(
        "Select Category",
        classes,
        classes,
        on_change=reload_train_data,
        key='train_bbox_class'
    )
    with st.expander("See Dataframe"):
        train_dataframe_placeholder = st.empty()
        reload_train_data()
        if st.session_state.train_dataframe is not None:
            train_dataframe_placeholder.dataframe(st.session_state.train_dataframe)

    col1, col2 = st.columns([1, 5])
    col1.number_input('Image ID', step=1, value=st.session_state.selected_train_image_id,
                      on_change=update_numin, key='number_input_image_id')
    col2.slider('Image ID', min_value=0, max_value=len(st.session_state.train_raw_dataset.dataset['images']) - 1,
                value=st.session_state.selected_train_image_id,
                on_change=update_slider, key='slider_image_id')
    train_image_placeholder = st.empty()
    draw_train_image()
    if st.session_state.train_bboxed_image is not None:
        train_image_placeholder.image(st.session_state.train_bboxed_image)
        with st.expander("See Annotations"):
            st.dataframe(st.session_state.train_anns)


with tab2:
    st.title('Validation Result EDA')
    st.markdown('### Data source')
    #st.session_state.work_dirs = st.text_input('Input working_directory (Valid )', st.session_state.work_dirs, key='val_work_dirs')
    #st.session_state.val_file = st.text_input('Input validation csv filename (Valid )', st.session_state.val_file, key='val_file')

    st.subheader('Confusion Matrix')
    st.markdown('### Data source')
    st.session_state.work_dirs = st.text_input('Input working_directory (Confusion Matrix )', st.session_state.work_dirs, key='val_cm_work_dirs')
    st.session_state.val_file = st.text_input('Input validation csv filename (Confusion Matrix )', st.session_state.val_file, key='val_cm_file')

    eval_targets = st.multiselect(
        "Select Evaluation Size (BBox)",
        ['Small', 'Medium', 'Large'],
        ['Medium', 'Large']
    )
    limit_cm_iou = st.slider('Select IoU', 0.05, 0.95, 0.5, 0.05)

    st.button('Draw', on_click=draw_confusion_matrix, type="primary")
    cm_placeholder = st.empty()
    if st.session_state.complete_pred and st.session_state.confusion_matrix_image:
        cm_placeholder.image(st.session_state.confusion_matrix_image)
    st.markdown('---')

    st.subheader('Boundary Box with Image')
    require_sort = st.checkbox('Sort mAP')
    limit_bbox_iou = st.slider('Select minimum IoU', 0.05, 0.95, 0.5, 0.05)
    bbox_targets = st.multiselect(
        "Select Category",
        classes,
        classes
    )
    st.button('Show', type="primary")
