from __future__ import division
from math import floor
import os


import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

try:
    from neoConnector import all_cells_with_n_hops_in_area, save_edges
except ImportError:
    from .neoConnector import all_cells_with_n_hops_in_area, save_edges

class_map = {'inflammatory': 0, 'lymphocyte' : 1, 'fibroblast and endothelial': 2,
               'epithelial': 3, 'apoptosis / civiatte body': 4}

annotation_paths = [ #Filename, (xmin, xmax, ymin, ymax), path to annotation
    ['N10_1_1', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_1_1_annotated.txt'],
    ['N10_1_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_1_2_annotated.txt'],
    ['N10_2_1', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_2_1_annotated.txt'],
    ['N10_2_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_2_2_annotated.txt'],
    ['N10_3_1', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_3_1_annotated.txt'],
    ['N10_3_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_3_2_annotated.txt'],
    ['N10_4_1', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_4_1_annotated.txt'],
    ['N10_4_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_4_2_annotated.txt'],
    ['N10_5_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_5_2_annotated.txt'],
    ['N10_6_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_6_2_annotated.txt'],
    ['N10_7_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_7_2_annotated.txt'],
    ['N10_7_3', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_7_3_annotated.txt'],
    ['N10_8_2', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_8_2_annotated.txt'],
    ['N10_8_3', (0, 2000, 0, 2000), 'annotations/N10_annotated/N10_8_3_annotated.txt'],
    ['P7_HE_Default_Extended_3_2', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_3_2.txt'],
    ['P7_HE_Default_Extended_4_2', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_4_2.txt'],
    ['P7_HE_Default_Extended_5_2', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_5_2.txt'],
    ['P13_1_1', (0, 2000, 0, 2000), 'annotations/P13_annotated/P13_1_1_annotated.txt'],
    ['P13_1_2', (0, 2000, 0, 2000), 'annotations/P13_annotated/P13_1_2_annotated.txt'],
    ['P13_2_2', (0, 2000, 0, 2000), 'annotations/P13_annotated/P13_2_2_annotated.txt'],
    ['P7_HE_Default_Extended_1_1', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_1_1.txt'],
    ['P7_HE_Default_Extended_2_1', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_2_1.txt'],
    ['P7_HE_Default_Extended_2_2', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_2_2.txt'],
    ['P7_HE_Default_Extended_3_1', (0, 2000, 0, 2000), 'annotations/P7_annotated/P7_HE_Default_Extended_3_1.txt'],
    ['P9_1_1', (0, 2000, 0, 2000), 'annotations/P9_annotated/P9_1_1_annotated.txt'],
    ['P9_2_1', (0, 2000, 0, 2000), 'annotations/P9_annotated/P9_2_1_annotated.txt'],
    ['P9_3_1', (0, 2000, 0, 2000), 'annotations/P9_annotated/P9_3_1_annotated.txt'],
    ['P9_4_1', (0, 2000, 0, 2000), 'annotations/P9_annotated/P9_4_1_annotated.txt'],
]


def create_edges():
    for path in annotation_paths:
        cells, dist = _read_from_db(path) # Get cells and AdjacencyMatrix
        points = parse_points(path[2]) # Get annotation path points
        coords = np.array([[cell.get('x'), cell.get('y')] for cell in cells]) # Get cell coordinates
        ids = np.array([cell.id for cell in cells]) # Get cell ids
        classes = np.array([class_map[cell.get('type')] for cell in cells]) # Get cell classes
        class_scores = np.array([np.array([cell.get('c0'), cell.get('c1'), cell.get('c2'), cell.get('c3')]) for cell in cells]) # Get cell classes

        dist = np.array(dist)
        adj = np.copy(dist)
        adj[adj != 0] = 1


        edges_all = get_intersections(points, coords, adj) # Calculate all edges passing over tissue layers
        print(path[0])
        save_edges(path[0]+'_final', edges_all, ids)

def _read_from_db(path):
    cells, adj = all_cells_with_n_hops_in_area(path[0]+"_final", path[1], hops=2)
    return cells, adj

def parse_points(fname):
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

if __name__ == '__main__':
    create_edges()
