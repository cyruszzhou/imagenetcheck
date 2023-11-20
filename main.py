import argparse
import os
import random
import shutil
import time
import warnings
import sys

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
# import torchvision.models as models
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import *
# import imagenet_seq

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', metavar='ARCH', default='resnet18')
parser.add_argument('--workers', default=20, type=int)
parser.add_argument('--epochs', default=180, type=int)
parser.add_argument('--job-epochs', default=180, type=int)
parser.add_argument('--start-epoch', default=0, type=int)
parser.add_argument('--batch-size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--print-freq', default=100, type=int)
parser.add_argument('--resume', default='auto', type=str)
parser.add_argument('--evaluate', dest='evaluate', action='store_true')
parser.add_argument('--pretrained', dest='pretrained', action='store_true')
parser.add_argument('--seed', default=None, type=int)

parser.add_argument('--p_init', type=float, default=8, help='value to initialize p with')
parser.add_argument('--lmbda', type=float, default=1e-08)
parser.add_argument('--job-id', type=str, default='default')
parser.add_argument('--ngpu', type=int, default=4)

parser.add_argument('--spm', type=str, default='None', help='"None", "pattern_4", "pattern_45"')

args = parser.parse_args()

args.save_path = './snapshots/' + args.job_id

best_acc1 = 0

def normalize_images(x):
        mu = 2 * torch.tensor([0.406, 0.456, 0.485]) - 1
        std = 2 * torch.tensor([0.225, 0.224, 0.229])
        mu = mu.view(1, -1, 1, 1).to(x.device)
        std = std.view(1, -1, 1, 1).to(x.device)
        return (x-mu) / std

def main_worker(args):
    global best_acc1

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'resnet18':
        # print("args spm", args.spm)
        model = PreResNet(p_init=args.p_init, block_sizes=[2,2,2,2], bottleneck=False, spm=args.spm)
    elif args.arch == 'resnet50':
        model = PreResNet(p_init=args.p_init, block_sizes=[3,4,6,3], bottleneck=True, spm=args.spm)

    #model = torch.nn.DataParallel(model).cuda()
    model = torch.nn.DataParallel(model.cuda(), device_ids=list(range(args.ngpu)))

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([p.numel() for p in trainable_params])
    print("Number of parameters: {}".format(params))

    criterion = nn.CrossEntropyLoss().cuda()

    weight_params = []
    prec_params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            if m.affine:
                weight_params.append(m.weight)
                weight_params.append(m.bias)
        elif isinstance(m, NoisyConv2d) or isinstance(m, NoisyLinear):
            weight_params.append(m.weight)
            prec_params.append(m.weight_s)
            if m.bias is not None:
                weight_params.append(m.bias)
                prec_params.append(m.bias_s)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            weight_params.append(m.weight)
            if m.bias is not None:
                weight_params.append(m.bias)

    print("Optimizing {} weight params and {} precision params".format(len(weight_params), len(prec_params)))
    optimizers = []
    optimizers.append(torch.optim.Adam(prec_params, lr=1e-3))
    optimizers.append(torch.optim.SGD(weight_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay))

    if args.resume:
        if args.resume == 'auto':
            args.resume = os.path.join(args.save_path, 'checkpoint.pth.tar')
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizers[0].load_state_dict(checkpoint['optimizer0'])
            optimizers[1].load_state_dict(checkpoint['optimizer1'])
            print("=> loaded checkpoint '{}' accuracy={} (epoch {})" .format(args.resume, best_acc1, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    else:
        print("=> do not use any checkpoint")

    cudnn.benchmark = True

    # train_loader = imagenet_seq.data.Loader('train', batch_size=args.batch_size, fixup=False)
    # val_loader = imagenet_seq.data.Loader('val', batch_size=args.batch_size, fixup=False)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dir = '/root/research/imagenet/train/train'
    val_dir = '/root/research/imagenet/train/val'

    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    val_dataset = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            # transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            normalize,
        ]))



    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, sampler=None) # can add sampler if desired, pin_mem

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, sampler=None)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    args.stop_epoch = min(args.epochs, args.start_epoch + args.job_epochs)
    for epoch in range(args.start_epoch, args.stop_epoch):
        current_lr = adjust_learning_rate(optimizers, epoch, args)

        print('\n==>>[Epoch={:03d}/{:03d}] [lr={:6.8f} ] [Best : Accuracy={:.2f}] '.format(epoch, args.epochs, current_lr, best_acc1))

        train(train_loader, model, criterion, optimizers, epoch, args)

        print(model.module.print_precs())

        acc1 = validate(val_loader, model, criterion, args)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer0' : optimizers[0].state_dict(),
            'optimizer1' : optimizers[1].state_dict(),
        })

def train(train_loader, model, criterion, optimizers, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.train()

    end = time.time()
    for i, (inputs, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        #inputs = normalize_images(inputs)
        #inputs, targets = inputs.cuda(), targets.cuda()

        # changed from 180
        if epoch < 90:
            model.module.set_mode('noisy')
        else:
            model.module.set_mode('quant')

        output = model(inputs)
        loss = criterion(output, targets)

        # changed from 180
        if epoch < 90:
            loss += args.lmbda * model.module.prec_cost()

        acc1, acc5 = accuracy(output, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1[0], inputs.size(0))
        top5.update(acc5[0], inputs.size(0))

        for opt in optimizers:
            opt.zero_grad()

        loss.backward()

        for opt in optimizers:
            if not (isinstance(opt, torch.optim.Adam) and epoch > 180):
                opt.step()

        with torch.no_grad():
            model.module.project()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            #input = normalize_images(input)

            output = model(input)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def save_checkpoint(state):
    path1 = os.path.join(args.save_path, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    path2 = os.path.join(args.save_path, 'checkpoint.pth.tar')
    torch.save(state, path1)
    shutil.copyfile(path1, path2)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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

def adjust_learning_rate(optimizers, epoch, args):
    decay_factor = .5
    if args.epochs == 180:
        steps = [45, 90, 128, 142, 158]
    elif args.epochs == 90:
        steps = [30, 60, 85]

    for step in steps:
        if epoch > step:
            decay_factor *= 10

    sgdlr = args.lr / decay_factor
    adamlr = 1e-3 / decay_factor
    for opt in optimizers:
        if isinstance(opt, torch.optim.Adam): lr = adamlr
        else: lr = sgdlr
        for param_group in opt.param_groups:
            param_group['lr'] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main_worker(args)