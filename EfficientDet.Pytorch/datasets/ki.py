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

        # Total size 547
        if set_name == 'train':
            self.image_set = images[:344] # 57*16 images
            self.target_set = target[:344]  # [[[xmin, ymin, xmax, ymax, label_ind], ... ], [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        elif set_name == 'val':
            self.image_set = images[344:430] # 57*16 images
            self.target_set = target[344:430]  # [[[xmin, ymin, xmax, ymax, label_ind], ... ], [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        elif set_name == 'test':
            self.image_set = images[430:] # 57*16 images
            self.target_set = target[430:]  # [[[xmin, ymin, xmax, ymax, label_ind], ... ], [[xmin, ymin, xmax, ymax, label_ind], ... ]]
        print(set_name, len(self.image_set))


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
