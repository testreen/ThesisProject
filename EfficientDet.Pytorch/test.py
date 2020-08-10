import torch
from models import EfficientDet
from models.efficientnet import EfficientNet
from datasets.parseKI import parseKI
from collections import OrderedDict
from datasets.visual_aug import visualize_bbox
import numpy as np
import cv2

if __name__ == '__main__':
    # Test load dataset
    image_set, target_set = parseKI(basePath="datasets/")

    print(len(image_set))

    for index in range(len(image_set)):
        print(index)
        target = target_set[index]
        img = image_set[index]

        target = np.array(target)

        bbox = target[:, :4]
        labels = target[:, 4]

        vis = visualize_bbox(img, bbox.tolist(), labels.tolist())
        cv2.imshow('image', vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Test load efficientnet model
    model = EfficientNet.from_pretrained('efficientnet-b0', advprop=False, num_classes=4)

    # Test load efficientnet checkpoint
    checkpoint = torch.load('models/model_kebnekaise.pth.tar', map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    checkpoint['state_dict'] = new_state_dict
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format('models/model_kebnekaise.pth.tar', checkpoint['epoch']))

    # Test EfficientNet
    inputs = torch.randn(4, 3, 512, 512)
    P = model(inputs)
    for idx, p in enumerate(P):
        print('P{}: {}'.format(idx, p.size()))

    # print('model: ', model)

    # Test efficientdet inference
    model = EfficientDet(num_classes=4, is_training=False)
    output = model(inputs)
    for out in output:
        print(out.size())
        print(out)
