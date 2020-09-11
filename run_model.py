import argparse
import os
import random
import shutil
import time
import warnings
import PIL

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from sklearn.metrics import classification_report


from efficientnet_pytorch import EfficientNet
from CellDataset import CellDataset

best_acc1 = 0

# Static config
num_classes = 4
class_names = ['inflammatory', 'lymphocyte', 'fibroblast and endothelial',
               'epithelial', 'apoptosis / civiatte body']
shuffle = True
k = 10  # Cross-validation splits

'''
Train/validate an EfficentNet model
'''
def run_model(loaders, split, args):
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
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    ngpus = torch.cuda.device_count()
    print("Available GPUs: {}".format(ngpus))

    return main_worker(loaders, split, args.gpu, ngpus, args)


def main_worker(loaders, split, gpu, ngpus, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    if 'efficientnet-b' in args.arch:  # NEW
        if args.pretrained:
            model = EfficientNet.from_pretrained(args.arch, advprop=args.advprop, num_classes=num_classes)
            print("=> using pre-trained model '{}'".format(args.arch))
        else:
            print("=> creating model '{}'".format(args.arch))
            model = EfficientNet.from_name(args.arch, override_params={'num_classes': num_classes})

    else:
        print("Only EfficientNet models are supported.")
        quit()

    # If specific GPU
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # Check if CUDA is available
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        else:
            model.to("cpu")

    # Class weights = (total_count - class_count) / total_count
    weights = [25137/2017, 25137/3211, 25137/7296, 25137/12519]
    weights = [(25137-2017)/25137, (25137-3211)/25137, (25137-7296)/25137, (25137-12519)/25137]
    class_weights = torch.FloatTensor(weights)
    if torch.cuda.is_available():
        class_weights = class_weights.cuda()

    # Define loss function (criterion) and optimizer
    if torch.cuda.is_available():
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda(args.gpu)
        if args.upsample:
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        if args.upsample:
            criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                nesterov=True,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #optimizer = torch.optim.AdamW(model.parameters(), args.lr,weight_decay=args.weight_decay, amsgrad=True)
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Only unfreeze classification layer
    if args.feature_extract:
        c = 0
        for param in model.parameters():
            c+= 1
            if c < 213:
                param.requires_grad = False

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(loaders['train'], model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1, _ = validate(loaders['val'], model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, split)

    if args.validate:
        res, classification = validate(loaders['val'], model, criterion, args)
        with open('res_val_{}.txt'.format(split), 'w') as f:
            print(res, file=f)
            print(classification, file=f)

    if args.evaluate:
        res, classification = validate(loaders['test'], model, criterion, args)
        with open('res_test_{}.txt'.format(split), 'w') as f:
            print(res, file=f)
            print(classification, file=f)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                              prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    topC1 = AverageMeter('Acc C1', ':6.2f')
    topC2 = AverageMeter('Acc C2', ':6.2f')
    topC3 = AverageMeter('Acc C3', ':6.2f')
    topC4 = AverageMeter('Acc C4', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, topC1, topC2, topC3, topC4,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))

            _, pred = output.topk(1, 1, True, True)
            pred = pred.t()

            y_true.extend(target.detach().cpu().numpy())
            y_pred.extend(pred.detach().cpu().numpy()[0])

            C1indices = [index for index, element in enumerate(target) if element == 0]
            if len(C1indices) > 0:
                accC1 = accuracy(output[C1indices], target[C1indices], topk=(1,))
                topC1.update(accC1[0].item(), len(C1indices))

            C2indices = [index for index, element in enumerate(target) if element == 1]
            if len(C2indices) > 0:
                accC2 = accuracy(output[C2indices], target[C2indices], topk=(1,))
                topC2.update(accC2[0].item(), len(C2indices))

            C3indices = [index for index, element in enumerate(target) if element == 2]
            if len(C3indices) > 0:
                accC3 = accuracy(output[C3indices], target[C3indices], topk=(1,))
                topC3.update(accC3[0].item(), len(C3indices))

            C4indices = [index for index, element in enumerate(target) if element == 3]
            if len(C4indices) > 0:
                accC4 = accuracy(output[C4indices], target[C4indices], topk=(1,))
                topC4.update(accC4[0].item(), len(C4indices))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc {top1.avg:.3f}'
              .format(top1=top1))

    report = classification_report(y_true, y_pred)
    print(report)
    return top1.avg, report


def save_checkpoint(state, is_best, split, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_{}.pth.tar'.format(split))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 3% every 2.4 epochs"""
    lr = args.lr * (0.97 ** (epoch // 2.4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

