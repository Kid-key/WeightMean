import argparse
import os
import random
import shutil
import time
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
from datetime import datetime
CHECKPOINT_PATH = 'checkpoint'
LOG_DIR = 'runs'
TIME_NOW = datetime.now().isoformat()
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

from data_loader import data_sets
from helper import AverageMeter, save_checkpoint, accuracy

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def linear_learning_rate(optimizer, epoch, init_lr,T_max=100):
    lr = 2e-5 - init_lr/T_max*(epoch+1-T_max)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
def consine_learning_rate(optimizer, epoch, init_lr,T_max=148):
 #   """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 2e-5 + init_lr*(1+math.cos(math.pi*epoch/T_max))/2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
# import imagenet_seq
# ImagenetLoader=imagenet_seq.data.Loader

model_names = [
    'vgg11_bn','vgg16_bn','vgg16_nobn','vgg16_wm','resnet18_wm', 'resnet50_wm', 'resnet101_wm',
    'resnet18', 'resnet50', 'resnet101', 'shufflenetv2', 'shufflenetv2_wm','mobilenetv2', 'mobilenetv2_wm'
]

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a','--arch', metavar='ARCH', default='alexnet', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
#parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_true',
                 #   help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print_freq', '-f', default=500, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoitn, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--path', default='', type=str, metavar='PATH',
                    help='path to imagenet dataset')
parser.add_argument('-mark', type=str, default='', help='remark to this experiment')
parser.add_argument('--warm', type=int, default=2, help='warm up training phase')
parser.add_argument('-record', type=bool, default=True, help='whether to save checkpoint and events')
parser.add_argument('--gpu', default = '0',help='GPU id')
parser.add_argument('--weight', default = 'none',help='none or mean')
parser.add_argument('--reg', default=0.05, type=float)
parser.add_argument('--schedular', type=str, default='')

parser.add_argument('--port', default='23496', type=str,
                    help='url used to set up distributed training')

print(TIME_NOW)
from torch.optim.lr_scheduler import _LRScheduler
class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (0.01+self.last_epoch / (self.total_iters + 1e-8)) for base_lr in self.base_lrs]

from models.vgg import vgg11_bn, vgg16_bn, vgg16_nobn

from models.resnet_nobn import resnet18_cbn, resnet50_cbn, resnet101_cbn,resnet18_nobn, resnet50_nobn, resnet101_nobn
from models.resnet_true import resnet18,  resnet50, resnet101


def meanweigh(module):
 #   for name, module in container.named_modules():
        if isinstance(module, nn.Conv2d) and module.groups==1:
            datavalue=module.weight.data
            #meanvalue=datavalue.mean([1,2,3],True)
            module.weight.data-=datavalue.mean([1,2,3],True)
        elif isinstance(module, nn.Linear):
            datavalue=module.weight.data
            #meanvalue=datavalue.mean([1],True)
            module.weight.data-=datavalue.mean([1],True)

import torch.nn.functional as F
class AdaptiveCrossEntropy(nn.Module):
    def __init__(self, reg: float = 0.2, reduction='mean'):
        super().__init__()
        self.reg = reg
        print('AdaptiveCrossEntropy with reg ', reg)

    def forward(self, preds, target):
        preds -= preds.mean([0])
        sqV = (preds**2).mean()
        log_preds = F.log_softmax(preds, dim=-1)
        loss = F.nll_loss(log_preds, target, reduction='mean')/torch.sqrt(sqV.detach())+sqV*self.reg
        return loss

best_prec1 = 0.0

def train(train_loader, model, criterion, optimizer, epoch,args,warmup_scheduler=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_freq = args.print_freq


    # switch to train mode
    model.train()
    #model.apply(freeze_bn)
    dslen=len(train_loader)
    end = time.time()
    for k,(input, target) in enumerate(train_loader):
        if epoch < args.warm:
            warmup_scheduler.step()

        # measure data loading time
        if True:
            data_time.update(time.time() - end)

            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)

            #prec1= accuracy(output.data, target, topk=(1, 5))
            loss = criterion(output, target)
            # measure accuracy and record loss
            prec= accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec[0].item(), input.size(0))
            top5.update(prec[1].item(), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        if 'wm' in args.arch:
            model.apply(meanweigh)#(model)

        n_iter = (epoch) * len(train_loader) + k

        if args.distributed and args.gpu != 0:
            continue
        if n_iter%print_freq==1 and epoch<args.start_epoch+5:   
            print('Epoch: [{0}][{1}/{2}]\t'
                   'LR: {3:.5f}\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'LossEpoch {loss.avg:.4f}\t'
                      'Prec@1 {top1.avg:.3f}\t'
                      'Prec@5 {top5.avg:.3f}\t'.format(
                    epoch, n_iter,dslen, optimizer.param_groups[0]['lr'],batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)
            batch_time.reset()
            data_time.reset()
            losses.reset()
            top1.reset()
            top5.reset()
            #validatetrain(val_loader, model, criterion)
        elif k==dslen-1:
            model.apply(inspect_bn)
            print('Epoch: [{0}][{1}/{2}]\t'
                   'LR: {3:.5f}\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'LossEpoch {loss.avg:.4f}\t'
                      'Precp@1 {top1.avg:.3f}\t'
                      'Preca@1 {top5.avg:.3f}\t'.format(
                    epoch, n_iter,dslen, optimizer.param_groups[0]['lr'],batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1, top5=top5), flush=True)


def freeze_bn(m):
    if isinstance(m, nn.BatchNorm1d):
       # m.train()
        m.track_running_stats=False

def inspect_bn(m):
    if isinstance(m, nn.BatchNorm1d):
       # m.train()
        print(m.running_var.data[:10])

#model.apply(freeze_bn)
def validate(val_loader, model, criterion, epoch,args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    print_freq = args.print_freq

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            prec1 = accuracy(output.data, target, topk=(1, 5))
            loss = criterion(output, target)
            prec2 = accuracy(output.data, target, topk=(1, 5))

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].item(), input.size(0))
            top5.update(prec2[0].item(), input.size(0))

            # measure elapsed time
            if args.distributed and args.gpu != 0:
                continue
            if i % (len(val_loader)//2) == 20:
                print('Test: [{0}][{1}/{2}]\t'
                      'Precp@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Preca@1 {top5.val:.3f} ({top5.avg:.3f})'.format(epoch,
                    i, len(val_loader),  loss=losses,
                    top1=top1, top5=top5))
        print(' * Accp@1 {top1.avg:.3f} Acca@1 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    return top1.avg, top5.avg


checkpoint_path = os.path.join(CHECKPOINT_PATH,TIME_NOW)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


def main():
    args = parser.parse_args()
    # if args.gpu!='-1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # if args.gpu!='-1':
    #     os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    args.distributed = ',' in args.gpu

    if args.distributed:
        args.dist_url = 'tcp://127.0.0.1:'+args.port
        gpus = list(map(int,args.gpu.split(',')))
        n_gpu = len(gpus)
        args.world_size = n_gpu
        mp.spawn(main_worker, nprocs=n_gpu, args=(n_gpu, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, 1, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_prec1
    args.gpu = gpu
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.dist_url,
                                world_size=args.world_size, rank=args.gpu)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))

    if args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    # elif args.arch == 'vgg11_wm':
    #     model = vgg11_nobn(2.8)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn()
    elif args.arch == 'vgg16_nobn':
        model = vgg16_nobn(2.0)
    elif args.arch == 'vgg16_wm':
        model = vgg16_nobn(2.8)
    # elif args.arch == 'vgg19_bn':
    #     model = vgg19_bn()
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet18_cbn':
        model = resnet18_cbn()
    elif args.arch == 'resnet50_cbn':
        model = resnet50_cbn()
    elif args.arch == 'resnet101_cbn':
        model = resnet101_cbn()
    elif args.arch == 'resnet18_wm':
        model = resnet18_nobn()
    elif args.arch == 'resnet50_wm':
        model = resnet50_nobn()
    elif args.arch == 'resnet101_wm':
        model = resnet101_nobn()
    elif args.arch == 'shufflenetv2':
        from models.shufflenetv2 import shufflenet_v2_x0_5
        model = shufflenet_v2_x0_5(pretrained=args.pretrained)
    elif args.arch == 'shufflenetv2_wm':
        from models.shufflenetv2_nobn import shufflenet_v2_x0_5
        model = shufflenet_v2_x0_5(cbn=False)
    elif args.arch == 'mobilenetv2':
        from models.mobilenetv2 import mobilenet_v2
        model = mobilenet_v2(pretrained=args.pretrained)
    elif args.arch == 'mobilenetv2_wm':
        from models.mobilenetv2_nobn import mobilenet_v2
        model = mobilenet_v2(cbn=False)
    else:
        raise NotImplementedError
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    else:
        args.gpu = int(args.gpu)
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
        #     model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    if args.reg>=0:
        criterion = AdaptiveCrossEntropy(reg=args.reg).cuda(args.gpu) #
    else:
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = optim.SGD(model.parameters(), lr=args.lr * args.batch_size / 256.,
                          momentum=args.momentum,nesterov=False,
                          weight_decay=args.weight_decay)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,85], gamma=0.1) #learning rate decay


    train_dataset, val_dataset = data_sets(args.path)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    # optionlly resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            warmup_scheduler = None
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if 'para' in checkpoint.keys():
                optstate=checkpoint['opt']
                checkpoint=checkpoint['para']
                optimizer.load_state_dict(optstate)
                print("=> loaded optimizer from '%s' ", args.resume)
            model.load_state_dict(checkpoint)
            parstr=args.resume.split('-')
            if args.start_epoch==0:
                args.start_epoch =int(parstr[-2])+1
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, int(parstr[-2])))
    else:
        iter_per_epoch = len(train_loader)
        warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch*args.warm )
        print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if 'wm' in args.arch:
        print('weight mean')
        model.apply(meanweigh)#(model)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return
    lastepoch = 0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if epoch >= args.warm:
            if 'cos' in args.schedular:
                consine_learning_rate(optimizer, epoch, args.lr, T_max=args.epochs - 1)
            elif 'lin' in args.schedular:
                linear_learning_rate(optimizer, epoch, args.lr, T_max=args.epochs)
            else:
                train_scheduler.step(epoch)
        train(train_loader, model, criterion, optimizer, epoch, args, warmup_scheduler)
        precb1, precp1 = validate(val_loader, model, criterion, epoch,args)
        prec1 = max(precb1,precp1)
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        if not args.distributed or (args.distributed and args.gpu == 0):
            if epoch > 80 and is_best:
                # if lastepoch>0:
                #     os.remove(checkpoint_path.format(net=args.arch, epoch=lastepoch, type='best'))
                torch.save({'para':model.state_dict(),'opt':optimizer.state_dict()}, checkpoint_path.format(net=args.arch, epoch=epoch, type='regular'))
                lastepoch=epoch
    print('Best Accuracy is {acc} in {epoch}'.format(acc=best_prec1,epoch=lastepoch))

if __name__ == '__main__':
    main()
