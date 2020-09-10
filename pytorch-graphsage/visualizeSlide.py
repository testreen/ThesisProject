from math import ceil
import json
import os
import sys
from PIL import Image
import cv2

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import link_prediction
from layers import MeanAggregator, LSTMAggregator, MaxPoolAggregator, MeanPoolAggregator
import models
import utils

'''
#test dataset
annotation_paths = [ #Filename, Coordinates, path to annotation
    ['P9_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_1_1_annotated.txt', 'datasets/images/P9/P9_1_1.tif'],
    ['P9_2_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_2_1_annotated.txt', 'datasets/images/P9/P9_2_1.tif'],
    ['P9_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_3_1_annotated.txt', 'datasets/images/P9/P9_3_1.tif'],
    ['P9_4_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_4_1_annotated.txt', 'datasets/images/P9/P9_4_1.tif'],
]
'''
'''
#validation dataset
annotation_paths = [
    ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt', 'datasets/images/P7/P7_HE_Default_Extended_1_1.tif'],
    ['P7_HE_Default_Extended_2_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_2_1.txt', 'datasets/images/P7/P7_HE_Default_Extended_2_1.tif'],
    ['P7_HE_Default_Extended_2_2', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_2_2.txt', 'datasets/images/P7/P7_HE_Default_Extended_2_2.tif'],
    ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt', 'datasets/images/P7/P7_HE_Default_Extended_3_1.tif'],
]
'''
'''
#train dataset
annotation_paths = [
    ['P13_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_1_1_annotated.txt', 'datasets/images/P13/P13_1_1.tif'],
    ['P13_2_2', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_2_2_annotated.txt', 'datasets/images/P13/P13_2_2.tif'],
    ['N10_1_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_1_2_annotated.txt', 'datasets/images/N10/N10_1_2.tif'],
    ['N10_4_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_4_2_annotated.txt', 'datasets/images/N10/N10_4_2.tif'],
    ['N10_5_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_5_2_annotated.txt', 'datasets/images/N10/N10_5_2.tif'],
    ['N10_6_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_6_2_annotated.txt', 'datasets/images/N10/N10_6_2.tif'],
]
'''
'''

'''


annotation_paths = [
    ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt', 'datasets/images/P7/P7_HE_Default_Extended_1_1.tif'],
    ['P7_HE_Default_Extended_2_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_2_1.txt', 'datasets/images/P7/P7_HE_Default_Extended_2_1.tif'],
    ['P7_HE_Default_Extended_2_2', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_2_2.txt', 'datasets/images/P7/P7_HE_Default_Extended_2_2.tif'],
    ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt', 'datasets/images/P7/P7_HE_Default_Extended_3_1.tif'],
    ['P13_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_1_1_annotated.txt', 'datasets/images/P13/P13_1_1.tif'],
    ['P13_2_2', (0, 2000, 0, 2000), 'datasets/annotations/P13_annotated/P13_2_2_annotated.txt', 'datasets/images/P13/P13_2_2.tif'],
    ['N10_1_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_1_2_annotated.txt', 'datasets/images/N10/N10_1_2.tif'],
    ['N10_4_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_4_2_annotated.txt', 'datasets/images/N10/N10_4_2.tif'],
    ['N10_5_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_5_2_annotated.txt', 'datasets/images/N10/N10_5_2.tif'],
    ['N10_6_2', (0, 2000, 0, 2000), 'datasets/annotations/N10_annotated/N10_6_2_annotated.txt', 'datasets/images/N10/N10_6_2.tif'],


    ['P9_1_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_1_1_annotated.txt', 'datasets/images/P9/P9_1_1.tif'],
    ['P9_2_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_2_1_annotated.txt', 'datasets/images/P9/P9_2_1.tif'],
    ['P9_3_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_3_1_annotated.txt', 'datasets/images/P9/P9_3_1.tif'],
    ['P9_4_1', (0, 2000, 0, 2000), 'datasets/annotations/P9_annotated/P9_4_1_annotated.txt', 'datasets/images/P9/P9_4_1.tif'],

]

def main(config, path):

    # Set up arguments for datasets, models and training.

    config['num_layers'] = len(config['hidden_dims']) + 1

    if config['cuda'] and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    config['device'] = device

    dataset_args = ('test', config['num_layers'])
    datasets = utils.get_dataset(dataset_args, path)
    loaders = []
    for i in range(len(datasets)):
        loaders.append(DataLoader(dataset=datasets[i], batch_size=config['batch_size'],
                            shuffle=False, collate_fn=datasets[i].collate_wrapper))
    input_dim, output_dim = datasets[0].get_dims()

    agg_class = utils.get_agg_class(config['agg_class'])
    model = models.GraphSAGE(input_dim, config['hidden_dims'],
                            output_dim, config['dropout'],
                            agg_class, config['num_samples'],
                            config['device'])
    model.to(config['device'])
    print(model)

    stats_per_batch = config['stats_per_batch']


    sigmoid = nn.Sigmoid()

    # Evaluate on test set.

    directory = os.path.join(os.path.dirname(os.getcwd()),
                            'trained_models')
    fname = utils.get_fname(config)
    model_path = os.path.join(directory, fname)
    model.load_state_dict(torch.load(model_path))
    criterion = utils.get_criterion(config['task'])
    stats_per_batch = config['stats_per_batch']

    t = config['threshold']
    model.eval()
    print('--------------------------------')
    print('Computing ROC-AUC score for the test dataset after training.')
    running_loss, total_loss = 0.0, 0.0
    num_correct, num_examples = 0, 0
    total_correct, total_examples, total_batches = 0, 0, 0
    y_true, y_scores, y_pred = [], [], []
    edge_pred, neg_pred, classes, coords = [], [], None, None
    for i in range(len(datasets)):
        num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
        total_batches += num_batches
        coords, classes = datasets[i].get_coords_and_class()

        for (idx, batch) in enumerate(loaders[i]):
            edges, features, node_layers, mappings, rows, labels, dist = batch
            features, labels = features.to(device), labels.to(device)
            out = model(features, node_layers, mappings, rows, dist)
            all_pairs = torch.mm(out, out.t())
            all_pairs = sigmoid(all_pairs)
            scores = all_pairs[edges.T]
            loss = criterion(scores, labels.float())
            running_loss += loss.item()
            total_loss += loss.item()
            predictions = (scores >= t).long()
            preds = edges[torch.nonzero(predictions).detach().cpu().numpy(), :]
            neg_preds = edges[(predictions == 0).nonzero(), :]

            if len(preds) > 0:
                for pred in preds:
                    edge_pred.append([node_layers[-1][pred[0][0]],node_layers[-1][pred[0][1]]])

            if len(neg_preds) > 0:
                for pred in neg_preds:
                    neg_pred.append([node_layers[-1][pred[0][0]],node_layers[-1][pred[0][1]]])

            num_correct += torch.sum(predictions == labels.long()).item()
            total_correct += torch.sum(predictions == labels.long()).item()
            num_examples += len(labels)
            total_examples += len(labels)
            y_true.extend(labels.detach().cpu().numpy())
            y_scores.extend(scores.detach().cpu().numpy())
            y_pred.extend(predictions.detach().cpu().numpy())
            if (idx + 1) % stats_per_batch == 0:
                running_loss /= stats_per_batch
                accuracy = num_correct / num_examples
                print('    Batch {} / {}: loss {:.4f}, accuracy {:.4f}'.format(
                    idx+1, num_batches, running_loss, accuracy))
                if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
                    area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
                    print('    ROC-AUC score: {:.4f}'.format(area))
                running_loss = 0.0
                num_correct, num_examples = 0, 0

        running_loss = 0.0
        num_correct, num_examples = 0, 0
    im = Image.open(path[3])
    imarray = np.array(im, dtype=np.double)/255
    imarray[:,:,[0,2]] = imarray[:,:,[2,0]]

    vis = utils.visualize_edges(imarray, edge_pred, neg_pred, coords, classes, path[2])
    cv2.imshow('image', vis)
    cv2.waitKey(0)
    #cv2.imwrite('results/edges_all_path_savgol_norm_{}.png'.format(path[0]),vis*255)
    cv2.destroyAllWindows()

    total_loss /= total_batches
    total_accuracy = total_correct / total_examples
    print('Loss {:.4f}, accuracy {:.4f}'.format(total_loss, total_accuracy))
    y_true = np.array(y_true).flatten()
    y_scores = np.array(y_scores).flatten()
    y_pred = np.array(y_pred).flatten()
    report = classification_report(y_true, y_pred)
    area = roc_auc_score(y_true, y_scores)
    print('ROC-AUC score: {:.4f}'.format(area))
    print('Classification report\n', report)
    print('Finished testing.')
    print('--------------------------------')

if __name__ == '__main__':
    config = utils.parse_args()
    for path in annotation_paths:
        #im = cv2.imread(path[3])
        # calculate mean value from RGB channels and flatten to 1D array
        #vals = im.mean(axis=2).flatten()
        # plot histogram with 255 bins
        #plt.plot(range(3))
        #b, bins, patches = plt.hist(im[:,:,0].flatten(), 255)
        #b, bins, patches = plt.hist(im[:,:,1].flatten(), 255)
        #b, bins, patches = plt.hist(im[:,:,2].flatten(), 255)
        #b, bins, patches = plt.hist(vals, 255)
        #plt.xlim([0,254])
        #plt.show()
        main(config, path)
