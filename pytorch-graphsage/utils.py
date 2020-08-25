import argparse
import importlib
import json
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL

from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import UnivariateSpline
from scipy.signal import savgol_filter

import torch
import torch.nn as nn

import networkx as nx

from tsp_solver.greedy import solve_tsp

from sklearn.neighbors import NearestNeighbors

from datasets import link_prediction
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
import models



# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

BOX_COLOR = [(1, 0, 0), (1, 1, 1), (0, 1, 0), (0, 0, 1)]
TEXT_COLOR = [(255, 255, 255),(0, 0, 0),(255, 255, 255),(255, 255, 255)]
KI_CLASSES = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
              'epithelial']


def visualize_edges(img, edges, neg_edges, coords, classes, path, class_idx_to_name=None, color=BOX_COLOR, thickness=1):
    img2 = img.copy()

    edge_centers = np.zeros((len(edges), 2), dtype=int)

    for i in range(len(edges)):
        id_1 = edges[i][0]
        id_2 = edges[i][1]

        x_1, y_1 = coords[id_1]
        x_2, y_2 = coords[id_2]

        edge_centers[i,0] = int((x_1 + x_2)/2)
        edge_centers[i,1] = int((y_1 + y_2)/2)

        cv2.line(img2, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 0, 0), thickness=2, lineType=8)

    for i in range(len(neg_edges)):
        id_1 = neg_edges[i][0]
        id_2 = neg_edges[i][1]

        x_1, y_1 = coords[id_1]
        x_2, y_2 = coords[id_2]

        cv2.line(img2, (int(x_1), int(y_1)), (int(x_2), int(y_2)), (0, 0, 1), thickness=1, lineType=8)

    edge_centers = np.c_[edge_centers[:,0], edge_centers[:,1]]

    # Get distance matrix and solve tsp
    dist_array = pdist(edge_centers)
    dist_matrix = squareform(dist_array)

    tsp_path = solve_tsp( dist_matrix.tolist())

    # Smooth path
    smooth_path = []
    avg_points = 15
    distance_max = 100
    for i in range(len(tsp_path)-avg_points):
        # stop at large jumps
        skip = avg_points
        for j in range(avg_points-1):
            if np.linalg.norm(edge_centers[tsp_path[i+j],:] - edge_centers[tsp_path[i+j+1],:]) > distance_max:
                skip = j
                break

        # filter out short paths
        if skip > 4:
            avgX = np.mean(edge_centers[tsp_path[i:i+skip],0])
            avgY = np.mean(edge_centers[tsp_path[i:i+skip],1])
            smooth_path.append(np.array((avgX, avgY)))

    split_paths = [[]]

    for i in range(len(smooth_path) - 1):
        dist = np.linalg.norm(smooth_path[i]-smooth_path[i+1])
        if (dist < distance_max):
            split_paths[-1].append(smooth_path[i])
            #cv2.line(img2, (int(smooth_path[i][0]), int(smooth_path[i][1])), (int(smooth_path[i+1][0]), int(smooth_path[i+1][1])), (1, 1, 1), thickness=2, lineType=8)
        else:
            split_paths.append([])

    # Savgol filter smoothing
    for paths in split_paths:
        if len(paths) > 3:
            paths = np.asarray(paths)
            x_smooth = savgol_filter(paths[:,0],min(15,(paths.shape[0] // 2) * 2 - 1), 2)
            y_smooth = savgol_filter(paths[:,1],min(15,(paths.shape[0] // 2) * 2 - 1), 2)
            for i in range(x_smooth.shape[0]-1):
                cv2.line(img2, (int(x_smooth[i]), int(y_smooth[i])), (int(x_smooth[i+1]), int(y_smooth[i+1])), (1, 1, 1), thickness=2, lineType=8)


    with open(path, 'r') as f:
        lines = f.readlines()
    points = [line[:-1].split(',') for line in lines] # Remove \n from line
    for k in range(len(points)-2):
        if len(points[k]) == 2 and len(points[k+1]) == 2:
            cv2.line(img2, (int(float(points[k][0])), int(float(points[k][1]))), (int(float(points[k+1][0])), int(float(points[k+1][1]))), (0, 1, 0), thickness=2, lineType=8)

    for i in range(len(coords)):
        x, y = coords[i]
        c = classes[i]
        #x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

        if int(c) != -1:
            cv2.rectangle(img2, (int(x)-2, int(y)-2), (int(x) + 2, int(y) + 2),
                          color=(color[int(c)]), thickness=thickness)

    for j in range(len(KI_CLASSES)):
        class_name = KI_CLASSES[j]
        ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.rectangle(img2, (0, 0 + j*int(1.3 * text_height)), (0 + text_width+3, 0 + (j+1)*int(1.3 * text_height)), BOX_COLOR[j], -1)
        cv2.putText(img2, class_name, (3, 0 + (j+1)*int(1.3 * text_height) - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 1,TEXT_COLOR[int(j)], lineType=cv2.LINE_AA)
    return img2

# Total 27, Test = 5, Train 18, Val 4

train_annotation_path = [ #Filename, (xmin, xmax, ymin, ymax), path to annotation
    ['N10_1_1', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_1_1_annotated.txt'],
    ['N10_1_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_1_2_annotated.txt'],
    ['N10_2_1', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_2_1_annotated.txt'],
    ['N10_2_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_2_2_annotated.txt'],
    ['N10_3_1', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_3_1_annotated.txt'],
    ['N10_3_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_3_2_annotated.txt'],
    ['N10_4_1', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_4_1_annotated.txt'],
    ['N10_4_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_4_2_annotated.txt'],
    ['N10_5_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_5_2_annotated.txt'],
    ['N10_6_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_6_2_annotated.txt'],
    ['N10_7_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_7_2_annotated.txt'],
    ['N10_7_3', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_7_3_annotated.txt'],
    ['N10_8_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_8_2_annotated.txt'],
    ['N10_8_3', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_8_3_annotated.txt'],
    ['P7_HE_Default_Extended_3_2', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_2.txt'],
    ['P7_HE_Default_Extended_4_2', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_4_2.txt'],
    ['P7_HE_Default_Extended_5_2', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_5_2.txt'],
    ['P13_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_1_1_annotated.txt'],
    ['P13_1_2', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_1_2_annotated.txt'],
    ['P13_2_2', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_2_2_annotated.txt'],
]

val_annotation_path = [ #Filename, Coordinates, path to annotation
    ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt'],
    ['P7_HE_Default_Extended_2_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_2_1.txt'],
    ['P7_HE_Default_Extended_2_2', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_2_2.txt'],
    ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt'],
]

test_annotation_path = [ #Filename, Coordinates, path to annotation
    ['P9_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_1_1_annotated.txt'],
    ['P9_2_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_2_1_annotated.txt'],
    ['P9_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_3_1_annotated.txt'],
    ['P9_4_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_4_1_annotated.txt'],
]

def get_agg_class(agg_class):
    """
    Parameters
    ----------
    agg_class : str
        Name of the aggregator class.
    Returns
    -------
    layers.Aggregator
        Aggregator class.
    """
    return getattr(sys.modules[__name__], agg_class)

def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.
    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'link_prediction':
        # Pos weight to balance dataset without oversampling
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([7.]))

    return criterion

def get_dataset(args, setPath=None):
    """
    Parameters
    ----------
    args : tuple
        Tuple of task, dataset name and other arguments required by the dataset constructor.
    setPath: list
        List of path data, example ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt']
    Returns
    -------
    dataset : torch.utils.data.Dataset
        The dataset.
    """
    datasets = []
    mode, num_layers = args
    if setPath == None:
        if mode == 'train':
            for path in train_annotation_path:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
                dataset = class_attr(path, mode, num_layers)
                datasets.append(dataset)
        elif mode == 'val':
            for path in val_annotation_path:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
                dataset = class_attr(path, mode, num_layers)
                datasets.append(dataset)
        elif mode == 'test':
            for path in test_annotation_path:
                class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
                dataset = class_attr(path, mode, num_layers)
                datasets.append(dataset)
    else:
        class_attr = getattr(importlib.import_module('datasets.link_prediction'), 'KIGraphDataset')
        dataset = class_attr(setPath, mode, num_layers)
        datasets.append(dataset)

    return datasets

def get_fname(config):
    """
    Parameters
    ----------
    config : dict
        A dictionary with all the arguments and flags.
    Returns
    -------
    fname : str
        The filename for the saved model.
    """
    agg_class = config['agg_class']
    hidden_dims_str = '_'.join([str(x) for x in config['hidden_dims']])
    num_samples = config['num_samples']
    batch_size = config['batch_size']
    epochs = config['epochs']
    lr = config['lr']
    weight_decay = config['weight_decay']
    fname = 'graphsage_agg_class_{}_hidden_dims_{}_num_samples_{}_batch_size_{}_epochs_{}_lr_{}_weight_decay_{}.pth'.format(
        agg_class, hidden_dims_str, num_samples, batch_size, epochs, lr,
        weight_decay)

    return fname


def parse_args():
    """
    Returns
    -------
    config : dict
        A dictionary with the required arguments and flags.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--json', type=str, default='config.json',
                        help='path to json file with arguments, default: config.json')

    parser.add_argument('--stats_per_batch', type=int, default=16,
                        help='print loss and accuracy after how many batches, default: 16')

    parser.add_argument('--dataset_path', type=str,
                        # required=True,
                        help='path to dataset')

    parser.add_argument('--task', type=str,
                        choices=['unsupervised', 'link_prediction'],
                        default='link_prediction',
                        help='type of task, default=link_prediction')

    parser.add_argument('--agg_class', type=str,
                        choices=[MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator],
                        default=MaxPoolAggregator,
                        help='aggregator class, default: MaxPoolAggregator')
    parser.add_argument('--cuda', action='store_true',
                        help='whether to use GPU, default: False')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout out, currently only for GCN, default: 0.5')
    parser.add_argument('--hidden_dims', type=int, nargs="*",
                        help='dimensions of hidden layers, length should be equal to num_layers, specify through config.json')
    parser.add_argument('--num_samples', type=int, default=-1,
                        help='number of neighbors to sample, default=-1')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='training batch size, default=32')
    parser.add_argument('--epochs', type=int, default=2,
                        help='number of training epochs, default=2')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate, default=1e-4')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay, default=5e-4')

    parser.add_argument('--save', action='store_true',
                        help='whether to save model in trained_models/ directory, default: False')
    parser.add_argument('--test', action='store_true',
                        help='load model from trained_models and run on test dataset')
    parser.add_argument('--val', action='store_true',
                        help='load model from trained_models and run on validation dataset')

    args = parser.parse_args()
    config = vars(args)
    if config['json']:
        with open(config['json']) as f:
            json_dict = json.load(f)
            config.update(json_dict)

            for (k, v) in config.items():
                if config[k] == 'True':
                    config[k] = True
                elif config[k] == 'False':
                    config[k] = False

    config['num_layers'] = len(config['hidden_dims']) + 1

    print('--------------------------------')
    print('Config:')
    for (k, v) in config.items():
        print("    '{}': '{}'".format(k, v))
    print('--------------------------------')

    return config
