import torch.utils.data as data
import numpy as np
from .parseKI import parseKI

KI_CLASSES = ('inflammatory', 'lymphocyte', 'fibroblast and endothelial',
              'epithelial')


class KiDataset(data.Dataset):
    """KI Detection Dataset Object
    input is image, target is annotation
    Arguments:
        root (string): filepath to KIdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
    """

    def __init__(self, root="", transform=None, set_name='train'):
        images, target = parseKI(basePath=root)

        # Total size 395
        if set_name == 'train':
            '''
            The count of inflammatory is: 1165
            The count of lymphocyte is: 2219
            The count of fibroblast and endothelial is: 4896
            The count of epithelial is: 8133
            '''
            self.image_set = images[:10] # 57*16 images
            self.target_set = target[:10]  # [[[xmin, ymin, xmax, ymax, label_ind], ... ], [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        elif set_name == 'val':
            '''
            The count of inflammatory is: 517
            The count of lymphocyte is: 572
            The count of fibroblast and endothelial is: 815
            The count of epithelial is: 1470
            '''
            self.image_set = images[256:265] # 57*16 images
            self.target_set = target[256:265]  # [[[xmin, ymin, xmax, ymax, label_ind], ... ], [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        elif set_name == 'test':
            '''
            The count of inflammatory is: 132
            The count of lymphocyte is: 204
            The count of fibroblast and endothelial is: 847
            The count of epithelial is: 2043
            '''
            self.image_set = images[320:] # 57*16 images
            self.target_set = target[320:]  # [[[xmin, ymin, xmax, ymax, label_ind], ... ], [[xmin, ymin, xmax, ymax, label_ind], ... ]]

        #counter = [0, 0, 0, 0]
        #for i in range(len(self.target_set)):
        #    for j in range(len(self.target_set[i])):
        #        counter[self.target_set[i][j][4]] += 1

        #for i in range(len(KI_CLASSES)):
        #    print('The count of {} is: {}'.format(KI_CLASSES[i], counter[i]))

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
