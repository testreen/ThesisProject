import argparse
import os
import random
import time
import warnings
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.efficientdet import EfficientDet
from datasets import Resizer, Normalizer, Augmenter, collater
from datasets.ki import KiDataset
from utils import EFFICIENTDET, get_state_dict
from eval import evaluate

parser = argparse.ArgumentParser(description='PyTorch EfficientDet Training')

parser.add_argument('--network', default='efficientdet-d0', type=str,
                    help='efficientdet-[d0, d1, ..]')
parser.add_argument('--dataset_root', default='datasets/', type=str,
                    help='path to dataset')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_epoch', default=500, type=int,
                    help='Num epoch for training')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_class', default=4, type=int,
                    help='Number of class used in model')
parser.add_argument('--device', default=[0, 1], type=list,
                    help='Use CUDA to train model')
parser.add_argument('--grad_accumulation_steps', default=1, type=int,
                    help='Number of gradient accumulation steps')
parser.add_argument('--lr', '--learning-rate', default=0.16, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=4e-5, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./saved/weights/', type=str,
                    help='Directory for saving checkpoint models')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

iteration = 1


def train(train_loader, model, scheduler, optimizer, epoch, args):
    global iteration
    print("{} epoch: \t start training....".format(epoch))
    start = time.time()
    total_loss = []
    model.train()
    model.module.is_training = True
    model.module.freeze_bn()
    optimizer.zero_grad()
    for idx, (images, annotations) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda().float()
            annotations = annotations.cuda()
        else:
            images = images.float()

        classification_loss, regression_loss = model([images, annotations])
        classification_loss = classification_loss.mean()
        regression_loss = regression_loss.mean()
        loss = classification_loss + regression_loss
        if bool(loss == 0):
            print('loss equal zero(0)')
            continue
        loss.backward()
        if (idx + 1) % args.grad_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad()

        total_loss.append(loss.item())
        if(iteration % 1 == 0):
            print('{} iteration: training ...'.format(iteration))
            ans = {
                'epoch': epoch,
                'iteration': iteration,
                'cls_loss': classification_loss.item(),
                'reg_loss': regression_loss.item(),
                'mean_loss': np.mean(total_loss)
            }
            for key, value in ans.items():
                print('    {:15s}: {}'.format(str(key), value))
        iteration += 1
    scheduler.step(np.mean(total_loss))
    result = {
        'time': time.time() - start,
        'loss': np.mean(total_loss)
    }
    for key, value in result.items():
        print('    {:15s}: {}'.format(str(key), value))


def test(dataset, model, epoch, args):
    print("{} epoch: \t start validation....".format(epoch))
    model = model
    model.eval()
    model.module.is_training = False
    with torch.no_grad():
        evaluate(dataset, model)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # Training dataset
    train_dataset = []
    train_dataset = KiDataset(
        root=args.dataset_root,
        set_name='train',
        transform=transforms.Compose(
            [
                Normalizer(),
                Augmenter()]))
    valid_dataset = KiDataset(
        root=args.dataset_root,
        set_name='val',
        transform=transforms.Compose(
            [
                Normalizer()]))
    #test_dataset = KiDataset(
    #    root=args.dataset_root,
    #    set_name='test',
    #    transform=transforms.Compose(
    #        [
    #            Normalizer()]))

    args.num_class = 4

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              num_workers=args.workers,
                              shuffle=True,
                              collate_fn=collater,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=1,
                              num_workers=args.workers,
                              shuffle=False,
                              collate_fn=collater,
                              pin_memory=True)

    checkpoint = []
    if(args.resume is not None):
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
        params = checkpoint['parser']
        args.num_class = params.num_class
        args.network = params.network
        args.start_epoch = checkpoint['epoch'] + 1
        del params

    model = EfficientDet(num_classes=args.num_class,
                         network=args.network,
                         W_bifpn=EFFICIENTDET[args.network]['W_bifpn'],
                         D_bifpn=EFFICIENTDET[args.network]['D_bifpn'],
                         D_class=EFFICIENTDET[args.network]['D_class']
                         )
    if(args.resume is not None):
        model.load_state_dict(checkpoint['state_dict'])
    del checkpoint

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        if torch.cuda.is_available():
            model = model.cuda()
            print('Run with DataParallel ....')
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)

    # define loss function (criterion) , optimizer, scheduler
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr)
    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.num_epoch):
        train(train_loader, model, scheduler, optimizer, epoch, args)

        if (epoch + 1) % 5 == 0:
            test(valid_dataset, model, epoch, args)

        state = {
            'epoch': epoch,
            'parser': args,
            'state_dict': get_state_dict(model)
        }

        torch.save(
            state,
            os.path.join(
                args.save_folder,
                args.network,
                "checkpoint_{}.pth".format(epoch)))


def main():
    args = parser.parse_args()
    if(not os.path.exists(os.path.join(args.save_folder, args.network))):
        os.makedirs(os.path.join(args.save_folder, args.network))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


if __name__ == "__main__":
    main()
