import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body', 'w']


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Run the model and predict label
def visualize_model(model, dataloader, num_images=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    was_training = model.training
    model.eval()
    images_so_far = 0
    plt.ioff()
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {},\n true: {}'.format(class_names[preds[j]], class_names[labels.numpy().astype(int)[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(mode=was_training)
