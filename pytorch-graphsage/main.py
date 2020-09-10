from math import ceil
import json
import os
import sys
import random

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

def main():

    # Set up arguments for datasets, models and training.
    config = utils.parse_args()
    config['num_layers'] = len(config['hidden_dims']) + 1

    if config['cuda'] and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    config['device'] = device

    # Get the dataset, dataloader and model.
    if not config['val'] and not config['test']:

        dataset_args = ('train', config['num_layers'])

    if config['val']:
        dataset_args = ('val', config['num_layers'])

    if config['test']:
        dataset_args = ('test', config['num_layers'])

    datasets = utils.get_dataset(dataset_args)
    loaders = []
    for i in range(len(datasets)):
        loaders.append(DataLoader(dataset=datasets[i], batch_size=config['batch_size'],
                        shuffle=True, collate_fn=datasets[i].collate_wrapper))
    input_dim, output_dim = datasets[0].get_dims()

    agg_class = utils.get_agg_class(config['agg_class'])
    model = models.GraphSAGE(input_dim, config['hidden_dims'],
                            output_dim, config['dropout'],
                            agg_class, config['num_samples'],
                            config['device'])
    model.to(config['device'])


    if config['val'] or config['test']:
        directory = os.path.join(os.path.dirname(os.getcwd()),
                                'trained_models')
        fname = utils.get_fname(config)
        path = os.path.join(directory, fname)
        model.load_state_dict(torch.load(path))

    print(model)

    stats_per_batch = config['stats_per_batch']

    criterion = utils.get_criterion(config['task'])

    sigmoid = nn.Sigmoid()

    # Compute ROC-AUC score for the untrained model.
    if not config['val'] and not config['test']:
        print('--------------------------------')
        print('Computing ROC-AUC score for the training dataset before training.')
        y_true, y_scores = [], []
        for i in range(len(datasets)):
            num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
            with torch.no_grad():
                for (idx, batch) in enumerate(loaders[i]):
                    edges, features, node_layers, mappings, rows, labels, dist = batch
                    features, labels = features.to(device), labels.to(device)
                    out = model(features, node_layers, mappings, rows, dist)
                    all_pairs = torch.mm(out, out.t())
                    all_pairs = sigmoid(all_pairs)
                    scores = all_pairs[edges.T]
                    y_true.extend(labels.detach().cpu().numpy())
                    y_scores.extend(scores.detach().cpu().numpy())
                    if (idx + 1) % stats_per_batch == 0:
                        print('    Batch {} / {}, Graph {} / {}'.format(idx+1, num_batches, i+1, len(datasets)))
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()
        area = roc_auc_score(y_true, y_scores)
        print('ROC-AUC score: {:.4f}'.format(area))
        print('--------------------------------')

    # Train.
    if not config['val'] and not config['test']:
        optimizer = optim.Adam(model.parameters(), lr=config['lr'],
                            weight_decay=config['weight_decay'])
        epochs = config['epochs']

        #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.8)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20, 25], gamma=0.5) # Epoch decay
        model.train()
        print('--------------------------------')
        print('Training.')
        for epoch in range(epochs):
            print('Epoch {} / {}'.format(epoch+1, epochs))

            epoch_loss = 0.0
            epoch_roc = 0.0
            epoch_batches = 0
            shuffle = list(range(len(loaders)))
            random.shuffle(shuffle) # Shuffle order of graphs
            for i in shuffle:
                num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
                epoch_batches += num_batches
                graph_roc = 0.0
                running_loss = 0.0
                for (idx, batch) in enumerate(loaders[i]):
                    edges, features, node_layers, mappings, rows, labels, dist = batch
                    features, labels = features.to(device), labels.to(device)
                    optimizer.zero_grad()
                    out = model(features, node_layers, mappings, rows, dist)
                    all_pairs = torch.mm(out, out.t())
                    all_pairs = sigmoid(all_pairs)
                    scores = all_pairs[edges.T]
                    loss = criterion(scores, labels.float())
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        running_loss += loss.item()
                        epoch_loss += loss.item()
                        if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
                            area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
                            epoch_roc += area
                            graph_roc += area
                running_loss /= num_batches
                print('    Graph {} / {}: loss {:.4f}'.format(
                    i+1, len(datasets), running_loss))
                print('    ROC-AUC score: {:.4f}'.format(graph_roc/num_batches))

            scheduler.step()
            print("Epoch avg loss: {}".format(epoch_loss / epoch_batches))
            print("Epoch avg ROC_AUC score: {}".format(epoch_roc / epoch_batches))

        print('Finished training.')
        print('--------------------------------')

    # Save trained model
    if not config['val'] and not config['test']:
        if config['save']:
            print('--------------------------------')
            directory = os.path.join(os.path.dirname(os.getcwd()),
                                    'trained_models')
            if not os.path.exists(directory):
                os.makedirs(directory)
            fname = utils.get_fname(config)
            path = os.path.join(directory, fname)
            print('Saving model at {}'.format(path))
            torch.save(model.state_dict(), path)
            print('Finished saving model.')
            print('--------------------------------')

    # Compute ROC-AUC score on validation set after training.
    if not config['val'] and not config['test']:
        print('--------------------------------')
        print('Computing ROC-AUC score for the validation dataset after training.')
        if not config['val']:
            dataset_args = ('val', config['num_layers'])
            datasets = utils.get_dataset(dataset_args)
            loaders = []
            for i in range(len(datasets)):
                loaders.append(DataLoader(dataset=datasets[i], batch_size=config['batch_size'],
                                    shuffle=False, collate_fn=datasets[i].collate_wrapper))
        y_true, y_scores = [], []
        for i in range(len(datasets)):
            num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
            with torch.no_grad():
                for (idx, batch) in enumerate(loaders[i]):
                    edges, features, node_layers, mappings, rows, labels, dist = batch
                    features, labels = features.to(device), labels.to(device)
                    out = model(features, node_layers, mappings, rows, dist)
                    all_pairs = torch.mm(out, out.t())
                    all_pairs = sigmoid(all_pairs)
                    scores = all_pairs[edges.T]
                    y_true.extend(labels.detach().cpu().numpy())
                    y_scores.extend(scores.detach().cpu().numpy())

                    if (idx + 1) % stats_per_batch == 0:
                        print('    Batch {} / {}, Graph {} / {}'.format(idx+1, num_batches, i+1, len(datasets)))
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()
        area = roc_auc_score(y_true, y_scores)
        print('ROC-AUC score: {:.4f}'.format(area))
        print('--------------------------------')

    # Plot the true positive rate and true negative rate vs threshold.
    if not config['val'] and not config['test']:
        tpr, fpr, thresholds = roc_curve(y_true, y_scores)
        tnr = 1 - fpr
        plt.plot(thresholds, tpr, label='tpr')
        plt.plot(thresholds, tnr, label='tnr')
        plt.xlabel('Threshold')
        plt.title('TPR / TNR vs Threshold')
        plt.legend()
        plt.show()

        # Choose an appropriate threshold and generate classification report on the validation set.
        idx1 = np.where(tpr <= tnr)[0]
        idx2 = np.where(tpr >= tnr)[0]
        t = thresholds[idx1[-1]]
        running_loss, total_loss = 0.0, 0.0
        num_correct, num_examples = 0, 0
        total_correct, total_examples, total_batches = 0, 0, 0
        y_true, y_scores, y_pred = [], [], []
        for i in range(len(datasets)):
            num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
            total_batches += num_batches
            with torch.no_grad():
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
                        print('    Batch {} / {}, Graph {} / {}: loss {:.4f}, accuracy {:.4f}'.format(
                            idx+1, num_batches, i+1, len(datasets), running_loss, accuracy))
                        if (torch.sum(labels.long() == 0).item() > 0) and (torch.sum(labels.long() == 1).item() > 0):
                            area = roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy())
                            print('    ROC-AUC score: {:.4f}'.format(area))
                        running_loss = 0.0
                        num_correct, num_examples = 0, 0
                running_loss = 0.0
                num_correct, num_examples = 0, 0
        total_loss /= total_batches
        total_accuracy = total_correct / total_examples
        print('Loss {:.4f}, accuracy {:.4f}'.format(total_loss, total_accuracy))
        print('Threshold: {:.4f}, accuracy: {:.4f}'.format(t, total_correct / total_examples))
        y_true = np.array(y_true).flatten()
        y_scores = np.array(y_scores).flatten()
        y_pred = np.array(y_pred).flatten()
        report = classification_report(y_true, y_pred)
        area = roc_auc_score(y_true, y_scores)
        print('ROC-AUC score: {:.4f}'.format(area))
        print('Classification report\n', report)
        print('Finished validating.')
        print('--------------------------------')

    if config['val']:
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
        for i in range(len(datasets)):
            num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
            total_batches += num_batches
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

        # Evaluate on test set.
    if config['test']:
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
        for i in range(len(datasets)):
            num_batches = int(ceil(len(datasets[i]) / config['batch_size']))
            total_batches += num_batches
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
    main()
