from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

label_paths = [
    'KI-Dataset/Training LabelIMG/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_1',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_2_2',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/Helena_P7/P7_HE_Default_Extended_3_1',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/N10/N10_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/N10/N10_1_2',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/N10/N10_1_3',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/N10/N10_2_1',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/N10/N10_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Helena/N10/N10_2_2',
    'KI-Dataset/Training LabelIMG/For KTH/Nikolce/N10_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Nikolce/N10_1_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_2_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_2_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_3_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_3_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_4_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P9/P9_4_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P13/P13_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P13/P13_1_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P13/P13_2_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P13/P13_2_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P19/P19_1_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P19/P19_1_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P19/P19_2_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P19/P19_2_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P19/P19_3_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P19/P19_3_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_1_3',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_1_4',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_2_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_2_3',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_2_4',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_3_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_3_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_3_3',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_4_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_4_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_4_3',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_5_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_5_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_6_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_6_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_7_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_7_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_8_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_8_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_9_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P20/P20_9_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P25/P25_2_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P25/P25_3_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P25/P25_3_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P25/P25_4_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P25/P25_5_1',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P25/P25_8_2',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P28/P28_10_4',
    'KI-Dataset/Training LabelIMG/For KTH/Rachael/Rach_P28/P28_10_5',
]

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']


def parseData(basePath="", fileCount=len(label_paths)):
    labels = []
    images = []

    for path in range(fileCount):
        # Parse xml tree
        tree = ET.parse(basePath+label_paths[path]+'.xml')
        root = tree.getroot()

        # Parse full image to nparray
        image = basePath+label_paths[path]+'.tif'
        im = Image.open(image)
        imarray = np.array(im, dtype=np.double)

        # Loop through crops
        for child in root.iter('object'):
            # Parse label
            for name in child.iter('name'):
                label = name.text
                labels.append(class_names.index(label))
                if label == 'w':
                    print(label_paths[path])
                    print(len(labels))

            # Parse matching image data
            for box in child.iter('bndbox'):
                boundaries = []
                for val in box.iter():
                    boundaries.append(val.text)

                # Get center of crop and make sure the box fits inside of image
                meanX = int((int(boundaries[1]) + int(boundaries[3])) / 2)
                meanY = int((int(boundaries[2]) + int(boundaries[4])) / 2)

                meanX = max(meanX, 18)
                meanX = min(meanX, imarray.shape[1]-18)

                meanY = max(meanY, 18)
                meanY = min(meanY, imarray.shape[0]-18)

                cropArray = imarray[meanY-18:meanY+18, meanX-18:meanX+18, :]/256
                images.append(cropArray)
    return images, labels
