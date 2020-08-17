from __future__ import division
from math import floor
import os


import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

try:
    from neoConnector import all_cells_with_n_hops_in_area
except ImportError:
    from .neoConnector import all_cells_with_n_hops_in_area

train_annotation_path = [ #Filename, Coordinates, path to annotation
    ['P7_HE_Default_Extended_1_1', (630, 1300, 350, 1200), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt']
]

val_annotation_path = [ #Filename, Coordinates, path to annotation
    ['P7_HE_Default_Extended_1_1', (1300, 1500, 1200, 1600), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt']
]

test_annotation_path = [ #Filename, Coordinates, path to annotation
    ['P7_HE_Default_Extended_1_1', (1500, 1700, 1600, 2000), 'datasets/annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt']
]

class_map = {'inflammatory': 0, 'lymphocyte' : 1, 'fibroblast and endothelial': 2,
               'epithelial': 3, 'apoptosis / civiatte body': 4}

class KIGraphDataset(Dataset):

    def __init__(self, path,  mode='train',
                 num_layers=2,
                 data_split=[0.8, 0.2]):
        """
        Parameters
        ----------
        path : str
            Path to the dataset file. For example, CollegeMsg.txt, etc.
        mode : str
            One of train, val or test. Default: train.
        num_layers : int
            Number of layers in the computation graph. Default: 2.
        data_split: list
            Fraction of edges to use for graph construction / train / val / test. Default: [0.85, 0.08, 0.02, 0.03].
        """
        super().__init__()

        if mode == 'train':
            self.path = train_annotation_path[path]
        elif mode == 'val':
            self.path = val_annotation_path[path]
        elif mode == 'test':
            self.path = test_annotation_path[path]


        self.mode = mode
        self.num_layers = num_layers
        self.data_split = data_split

        print('--------------------------------')
        print('Reading dataset from {}'.format(path))

        cells, adj = self._read_from_db(path) # Get cells and AdjacencyMatrix
        points = self.parse_points(self.path[2]) # Get annotation path points
        coords = np.array([[cell.get('x'), cell.get('y')] for cell in cells]) # Get cell coordinates
        classes = np.array([class_map[cell.get('type')] for cell in cells]) # Get cell classes
        adj_edge = np.array(adj_to_edge(adj)) # Get neighbors on edge format
        print(adj_edge.shape)
        return
        adj = np.array(adj) # Get neighbors on AdjacencyMatrix format

        edges_all = np.array(get_intersections(points, coords, adj)) # Calculate all edges passing between cells over path

        print('Finished reading data.')

        print('Setting up graph.')
        vertex_id = {j : i for (i, j) in enumerate(range(len(coords)))}

        idxs = [floor(v*edges_all.shape[0]) for v in np.cumsum(data_split)]

        edges_t, pos_examples = adj_edge, edges_all

        edges_t[:, :2] = np.array([vertex_id[u] for u in edges_t[:, :2].flatten()]).reshape(edges_t[:, :2].shape)
        edges_s = np.unique(edges_t[:, :2], axis=0)

        self.n = len(vertex_id) # Count vertices
        self.m_s, self.m_t = edges_s.shape[0], edges_t.shape[0] # Count edges

        adj = sp.coo_matrix((np.ones(self.m_s), (edges_s[:, 0], edges_s[:, 1])),
                            shape=(self.n,self.n),
                            dtype=np.float32)
        # Symmetric.
        adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        self.adj = adj.tolil()

        self.edges_s = edges_s # Edges between layers
        self.nbrs_s = self.adj.rows # Neighbors
        self.features = torch.from_numpy(np.eye(self.n)*classes).float() # Cell features

        self.nbrs_t = self.nbrs_s

        print('Finished setting up graph.')

        print('Setting up examples.')


        #pos_seen = set(tuple([u,v]) for (u,v) in edges_s)
        # Symmetric.
        #pos_seen |= set(tuple([v,u]) for (u,v) in edges_s)
        pos_examples = pos_examples[:, :2]
        #pos_examples = np.array([row for row in pos_examples \
        #                         if (row[0] < self.n) and
        #                         (row[1] < self.n) and
        #                         ((row[0], row[1]) not in pos_seen) and
        #                         ((row[1], row[0]) not in pos_seen)])

        pos_examples = np.unique(pos_examples, axis=0)

        # Generate negative examples not in cell edges crossing path
        num_neg_examples = pos_examples.shape[0]
        neg_examples = []
        cur = 0
        n, _choice = self.n, np.random.choice
        neg_seen = set(tuple(e[:2]) for e in edges_all)
        while cur < num_neg_examples:
            u, v = _choice(n, 2, replace=False)
            if (u, v) in neg_seen:
                continue
            cur += 1
            neg_examples.append([u, v])
        neg_examples = np.array(neg_examples, dtype=np.int64)

        # pos_examples, neg_examples = pos_examples[:1024], neg_examples[:1024]

        x = np.vstack((pos_examples, neg_examples))
        y = np.concatenate((np.ones(pos_examples.shape[0]),
                            np.zeros(neg_examples.shape[0])))
        perm = np.random.permutation(x.shape[0])
        x, y = x[perm, :], y[perm]
        x, y = torch.from_numpy(x).long(), torch.from_numpy(y).long()
        self.x, self.y = x, y

        print('Finished setting up examples.')

        print('Dataset properties:')
        print('Mode: {}'.format(self.mode))
        print('Number of vertices: {}'.format(self.n))
        print('Number of static edges: {}'.format(self.m_s))
        print('Number of temporal edges: {}'.format(self.m_t))
        print('Number of examples/datapoints: {}'.format(self.x.shape[0]))
        print('--------------------------------')


    def _read_from_db(self, path):
        cells, adj = all_cells_with_n_hops_in_area(self.path[0], self.path[1], hops=2)
        return cells, adj

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def _form_computation_graph(self, idx):
        """
        Parameters
        ----------
        idx : int or list
            Indices of the node for which the forward pass needs to be computed.
        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        """
        _list, _set = list, set
        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]

        for _ in range(self.num_layers):
            prev = node_layers[-1]
            arr = [node for node in prev]
            arr.extend([e for node in arr for e in self.nbrs_s[node]])  # add neighbors to graph
            arr = np.array(_list(_set(arr)), dtype=np.int64)
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i, j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset. An example is (edge, label).
        Returns
        -------
        edges : numpy array
            The edges in the batch.
        features : torch.FloatTensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            Each row is the list of neighbors of nodes in node_layers[0].
        labels : torch.LongTensor
            Labels (1 or 0) for the edges in the batch.
        """
        idx = list(set([v.item() for sample in batch for v in sample[0][:2]]))

        node_layers, mappings = self._form_computation_graph(idx)

        rows = self.nbrs_s[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = torch.FloatTensor([sample[1] for sample in batch])
        edges = np.array([sample[0].numpy() for sample in batch])
        edges = np.array([mappings[-1][v] for v in edges.flatten()]).reshape(edges.shape)

        # TODO: Pin memory. Change type of node_layers, mappings and rows to
        # tensor?

        return edges, features, node_layers, mappings, rows, labels

    def get_dims(self):
        return self.features.shape[0], 1

        return np.array(lines, dtype=np.int64)

    def parse_points(self, fname):
        with open(fname, 'r') as f:
            lines = f.readlines()
        lines = [line[:-1].split(',') for line in lines] # Remove \n from line
        return lines

def adj_to_edge(adj):
    edges = []
    for i in range(len(adj)):
        edges += ([[i,index] for index, element in enumerate(adj[i]) if element == 1])

    return edges

def get_intersections(points, coords, adj):
    # Loop through cells
    intersections = []
    count = 0
    for i in range(len(coords)):
        # Get ids of all neighbors
        nbrs = [index for index, element in enumerate(adj[i]) if element == 1]
        for j in range(len(nbrs)):
            passed = False
            for k in range(len(points)-2):
                if len(points[k]) == 2 and len(points[k+1]) == 2:
                    L1 = line(coords[i], coords[nbrs[j]]) # Line between node and neighbor
                    L2 = line([int(float(point)) for point in points[k]], [int(float(point)) for point in points[k+1]]) # Line between two points of path
                    inter = intersection(L1, L2) # Get x-coordinate for intersection or False if none
                    if inter != False:
                        if ( (inter > max( min(coords[i][0],coords[nbrs[j]][0]), min(int(float(points[k][0])),int(float(points[k+1][0]))) )) and
                            (inter < min( max(coords[i][0],coords[nbrs[j]][0]), max(int(float(points[k][0])),int(float(points[k+1][0]))) )) ): # If intersection is inside line segments
                            intersections.append([i, nbrs[j]])
                            passed = True
                            break
            #if passed == False: # If no intersections between cell and neighbor
            #    intersections.append([i, nbrs[j], 0])
            #
    return intersections

def line(p1, p2):
    A = (p1[1] - p2[1])
    B = (p2[0] - p1[0])
    C = (p1[0]*p2[1] - p2[0]*p1[1])
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    if D != 0:
        x = Dx / D
        return x
    else:
        return False
