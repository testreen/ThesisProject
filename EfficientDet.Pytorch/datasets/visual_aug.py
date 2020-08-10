import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL



# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = [(1, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 1)]
TEXT_COLOR = [(255, 255, 255),(0, 0, 0),(255, 255, 255),(255, 255, 255)]
KI_CLASSES = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
              'epithelial']


def visualize_bbox(img, bbox, class_id, class_idx_to_name=None, color=BOX_COLOR, thickness=1):
    img2 = img.copy()
    print(len(bbox), len(class_id))
    for i in range(len(bbox)):
        x_min, y_min, x_max, y_max = bbox[i]
        #x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        x_mean = (x_min + x_max) // 2
        y_mean = (y_min + y_max) // 2

        if int(class_id[i]) != -1:
            cv2.rectangle(img2, (int(x_mean)-2, int(y_mean)-2), (int(x_mean) + 2, int(y_mean) + 2),
                          color=(color[int(class_id[i])]), thickness=thickness)

    for j in range(len(KI_CLASSES)):
        class_name = KI_CLASSES[j]
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img2, (0, 0 + j*int(1.3 * text_height)), (0 + text_width+3, 0 + (j+1)*int(1.3 * text_height)), BOX_COLOR[j], -1)
        cv2.putText(img2, class_name, (3, 0 + (j+1)*int(1.3 * text_height) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 1,TEXT_COLOR[int(j)], lineType=cv2.LINE_AA)
    return img2


def visualize(img, bboxes, labels):
    img = visualize_bbox(
        img, bboxes, labels)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img)
    return img


def compare(img, bboxes, annotations):
    print(len(bboxes), len(annotations))
    for i in range(len(bboxes)):
        x_min, y_min, x_max, y_max = bboxes[i]
        #x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        x_mean = (x_min + x_max) // 2
        y_mean = (y_min + y_max) // 2
        if int(x_min) != -1:
            cv2.rectangle(img, (int(x_mean)-2, int(y_mean)-2), (int(x_mean) + 2, int(y_mean) + 2),
                          color=(1, 1, 1), thickness=1)

    for i in range(len(annotations)):
        x_min, y_min, x_max, y_max, label = annotations[i]
        #x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        x_mean = (x_min + x_max) // 2
        y_mean = (y_min + y_max) // 2
        if int(x_min) != -1:
            cv2.rectangle(img, (int(x_mean)-2, int(y_mean)-2), (int(x_mean) + 2, int(y_mean) + 2),
                          color=(0, 0, 0), thickness=1)

    return img


def visual_data(data, name):
    img = data['image']
    bboxes = data['bboxes']
    annotations = {'image': data['image'], 'bboxes': data['bboxes'], 'category_id': range(
        len(data['bboxes']))}
    category_id_to_name = {v: v for v in range(len(data['bboxes']))}

    img = visualize(annotations, category_id_to_name)
    cv2.imwrite(name, img)
