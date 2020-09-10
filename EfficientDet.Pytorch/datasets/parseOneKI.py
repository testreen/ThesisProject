from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET
import torch.utils.data as data

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']

KI_CLASSES = ('inflammatory', 'lymphocyte', 'fibroblast and endothelial',
              'epithelial')


class KiDataset(data.Dataset):
    """KI Detection Dataset Object for single slide
    input is image, target is annotation
    Arguments:
        root (string): filepath to KIdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
    """

    def __init__(self, root="", filePath="", transform=None):
        images, target, original_image, original_labels, normal_image = parseOneKI(basePath=root, filePath=filePath)

        self.image_set = images # 57*16 images
        self.target_set = target
        self.image = original_image
        self.targets = original_labels
        self.filePath = root+filePath
        self.filename = filePath.rsplit('/', 1)[-1]
        self.normal_image = normal_image

        self.transform = transform

    def __getitem__(self, index):
        target = self.target_set[index]
        img = self.image_set[index]

        target = np.array(target)

        sample = {'img': img, 'annot': target}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.image_set)

    def num_classes(self):
        return len(KI_CLASSES)

    def get_original_image(self, index):
        return self.image_set[index]

    def label_to_name(self, label):
        return KI_CLASSES[label]

    def load_annotations(self, index):
        return np.array(self.target_set[index])

mins = [0, 496, 992, 1488]
maxs = [512, 1008, 1504, 2000]

def translate_boxes(arr):
    #print(arr)
    labels = []
    for i in range(4):
        for j in range(4):
            index = i*4+j
            xmin = mins[i]
            ymin = mins[j]
            for k in range(len(arr[index])):
                label = []
                if(len(arr[index]) > 0):
                    label.append(arr[index][k][0]+xmin)
                    label.append(arr[index][k][1]+ymin)
                    label.append(arr[index][k][2]+xmin)
                    label.append(arr[index][k][3]+ymin)
                    labels.append(label)

    return labels




def parseOneKI(basePath="", filePath='KI-Dataset/For KTH/Rachael/Rach_P28/P28_10_5'):

    images = []
    labels = []
    original_labels = []

    # Parse full image to nparray
    image = basePath+filePath+'.tif'
    im = Image.open(image)
    imarray = np.array(im, dtype=np.double)/255



    # Pad image to 2000x2000x3
    padded_array = np.ones((2000, 2000, 3))
    shape = np.shape(imarray)
    padded_array[:shape[0], :shape[1]] = imarray
    imarray = padded_array

    normal = imarray.copy()
    B = imarray.copy()
    means = B.mean(axis=2)
    B[means > 0.98,:] = np.nan
    mean = np.nanmean(B, axis=(0,1))
    std = np.nanstd(B, axis=(0,1))
    imarray = (imarray - mean) / std

    # Parse xml tree if labels exist
    try:
        tree = ET.parse(basePath+filePath+'.xml')
        root = tree.getroot()
    except:
        targets = []
        slices = []
        for i in range(4):
            for j in range(4):
                xmin = mins[i]
                xmax = maxs[i]
                ymin = mins[j]
                ymax = maxs[j]
                #print(xmin, xmax, ymin, ymax, i*4+j)
                targets.append([])
                slices.append(imarray[ymin:ymax, xmin:xmax, :])
        for i in range(16):
            labels.append(targets[i])
            images.append(slices[i])
        return images, labels, imarray, original_labels, normal


    # Loop through crops
    for child in root.iter('object'):
        # Parse label
        target = [] # [xmin, ymin, xmax, ymax, label]
        label = ""
        for name in child.iter('name'):
            label = name.text

        # Parse matching image data
        for box in child.iter('bndbox'):
            boundaries = []
            for val in box.iter():
                boundaries.append(val.text)

            # Get center of crop and make sure the box fits inside of image
            meanX = int((int(boundaries[1]) + int(boundaries[3])) / 2)
            meanY = int((int(boundaries[2]) + int(boundaries[4])) / 2)

            meanX = max(meanX, 16)
            meanX = min(meanX, imarray.shape[1]-16)

            meanY = max(meanY, 16)
            meanY = min(meanY, imarray.shape[0]-16)

            # Check if full box is inside image
            if class_names.index(label) != 4:
                #print(meanX, xmin, xmax, meanY, ymin, ymax, i*4+j)
                target.append(max(meanX-16, 0))
                target.append(max(meanY-16, 0))
                target.append(min(meanX+16, 2000))
                target.append(min(meanY+16, 2000))
                target.append(class_names.index(label))
                original_labels.append(target)

    # Split into 512x512 slices of full image
    targets = []
    slices = []
    for i in range(4):
        for j in range(4):
            xmin = mins[i]
            xmax = maxs[i]
            ymin = mins[j]
            ymax = maxs[j]
            #print(xmin, xmax, ymin, ymax, i*4+j)
            targets.append([])
            slices.append(imarray[ymin:ymax, xmin:xmax, :])

            # Loop through crops
            for child in root.iter('object'):
                # Parse label
                target = [] # [xmin, ymin, xmax, ymax, label]
                label = ""
                for name in child.iter('name'):
                    label = name.text

                # Parse matching image data
                for box in child.iter('bndbox'):
                    boundaries = []
                    for val in box.iter():
                        boundaries.append(val.text)

                    # Get center of crop and make sure the box fits inside of image
                    meanX = int((int(boundaries[1]) + int(boundaries[3])) / 2)
                    meanY = int((int(boundaries[2]) + int(boundaries[4])) / 2)

                    meanX = max(meanX, 16)
                    meanX = min(meanX, imarray.shape[1]-16)

                    meanY = max(meanY, 16)
                    meanY = min(meanY, imarray.shape[0]-16)

                    # Check if full box is inside image
                    if meanX > xmin + 16 and meanX <= xmax - 16 and meanY > ymin + 16 and meanY <= ymax - 16 and class_names.index(label) != 4:
                        #print(meanX, xmin, xmax, meanY, ymin, ymax, i*4+j)
                        target.append(max(meanX-16-xmin, 0))
                        target.append(max(meanY-16-ymin, 0))
                        target.append(min(meanX+16-xmin, 512))
                        target.append(min(meanY+16-ymin, 512))
                        target.append(class_names.index(label))
                        targets[i*4+j].append(target)

    for i in range(16):
        labels.append(targets[i])
        images.append(slices[i])

    return images, labels, imarray, original_labels, normal
