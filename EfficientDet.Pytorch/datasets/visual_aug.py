import cv2
import matplotlib.pyplot as plt



# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = [(255, 0, 0), (0, 0, 0), (0, 255, 0), (0, 0, 255)]
TEXT_COLOR = (255, 255, 255)


def visualize_bbox(img, bbox, class_id, class_idx_to_name=None, color=BOX_COLOR, thickness=2):
    for i in range(len(bbox)):
        x_min, y_min, x_max, y_max = bbox[i]
        x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                      color=color[class_id[i]], thickness=thickness)
    # class_name = class_idx_to_name[class_id]
    # ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    # cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    # cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35,TEXT_COLOR, lineType=cv2.LINE_AA)
    return img


def visualize(img, bboxes, labels):
    img = visualize_bbox(
        img, bboxes.tolist(), labels)
    # plt.figure(figsize=(12, 12))
    # plt.imshow(img)
    return img


def visual_data(data, name):
    img = data['image']
    bboxes = data['bboxes']
    annotations = {'image': data['image'], 'bboxes': data['bboxes'], 'category_id': range(
        len(data['bboxes']))}
    category_id_to_name = {v: v for v in range(len(data['bboxes']))}

    img = visualize(annotations, category_id_to_name)
    cv2.imwrite(name, img)
