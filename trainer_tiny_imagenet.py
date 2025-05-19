import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet50 as resnet

# from torchsummary import summary

from datetime import datetime
from logzero import logger
import matplotlib.pyplot as plt
import numpy as np

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

# from autoattack import AutoAttack
from tiny_imagenet import load_tinyimagenet
import foolbox as fb

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))



parser = argparse.ArgumentParser(description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    # choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--warm_up', dest='warm_up',default=1, action='store_true', help='use warm up or not')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set',default=True)
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=25)
parser.add_argument('--eps', dest='eps',help='Input variation parameter',type=float, default=0.01)

best_prec1 = 0

def main():
    global args, best_prec1
    args = parser.parse_args()


    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    model = resnet.__dict__[args.arch]()
    model.cuda()
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_dataset , test_dataset = load_tinyimagenet("") # Add your dataset location here
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                                       num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                                                       num_workers=args.workers, pin_memory=True)
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if args.half:
        model.half()
        criterion.half()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1e-4, momentum=0.9)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs-args.warm_up*4, eta_min=0, last_epoch= -1, verbose=False)
     # Multistep
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                     milestones=[100, 150], last_epoch=args.start_epoch - 1)

    # Cosine 
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
    #                                         T_0 = 1,# Number of iterations for the first restart
    #                                         T_mult = 2, # A factor increases TiTi after a restart
    #                                         eta_min = 1e-4) # Minimum learning rate

    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
                                        base_lr = 0.01,# base lr
                                        max_lr = 0.1, # max_lr
                                        mode = "triangular",
                                        step_size_up=8000,
                                        ) # mode

    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.1,
    #                                           max_momentum=0, base_momentum=0,pct_start=0.45,
    #                                           steps_per_epoch=len(train_loader),epochs=args.epochs,
    #                                           anneal_strategy="linear",div_factor=10,
    #                                           three_phase=True)
    # eta_max=0.1
    # eta_min=0.07
    # offset=0.08
    # lr_scheduler = torch.optim.lr_scheduler.UniformNoisyLR(optimizer=optimizer,max_lr=eta_max,min_lr=eta_min,offset=offset) #gamma is the multiplicative factor
#steps per epoch=391, explore epochs=100=39100 steps, warmup is 10 percent of total steps, not needed for resnet
    # lr_scheduler=torch.optim.lr_scheduler.KneeLRScheduler(optimizer=optimizer,peak_lr = 0.1,warmup_steps= 0,explore_steps=39100,total_steps=195500)
    

    if args.arch in ['resnet1202', 'resnet110']:
        # for resnet1202 original paper uses lr=0.01 for first 400 minibatches for warm-up
        # then switch back. In this setup it will correspond for first epoch.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr*0.1

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))
        train(train_loader, model, criterion, optimizer, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        if is_best:
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint_proposed_tiny_imagenet.th'))

    if args.evaluate:
        filename = filename=os.path.join(args.save_dir, 'checkpoint_proposed_tiny_imagenet.th')
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        test(val_loader, model, criterion)
        return

def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))
    
            
def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()
            if args.half:
                input_var = input_var.half()
            # compute output
            output = model(input_var)# model prediction on clean examples
            output = output.float()
            prec1 = accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))
            loss = criterion(output, target_var)
            loss = loss.float()
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            i, len(val_loader), batch_time=batch_time, loss=losses,
                            top1=top1))
            
        print(' * Prec@1 {:.3f}'.format(top1.avg))
    return top1.avg

def test(test_loader, model, criterion):
    """
    Run evaluation
    """
    
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top_fgsm = AverageMeter()
    top_pgd = AverageMeter()
    top_aa = AverageMeter()
    # switch to evaluate mode
    model.eval()

    end = time.time()

    # with torch.no_grad():
    for i, (input, target) in enumerate(test_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()

        adversary = AutoAttack(model, norm='Linf', eps=1/255)
        x_aa = adversary.run_standard_evaluation(input_var, target_var)

        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        attack = fb.attacks.FGSM()
        raw_advs, x_fgm, success = attack(fmodel, input_var, criterion=fb.criteria.Misclassification(target_var),epsilons=[8/255])


        attack = fb.attacks.LinfPGD()
        _, x_pgd, _  = attack(fmodel, input_var ,criterion=fb.criteria.Misclassification(target_var), epsilons=[8/(255)])
        output = model(input_var)
        # compute output
        
        output_fgm = model(x_fgm[0]) 
        output_pgd = model(x_pgd[0]) 
        output_aa = model(x_aa)
        
        output = output.float()
        output_fgm = output_fgm.float()
        output_pgd = output_pgd.float()
        output_aa = output_aa.float()

        prec1 = accuracy(output.data, target)[0]
        prec1_fgm = accuracy(output_fgm.data, target)[0]
        prec1_pgd = accuracy(output_pgd.data, target)[0]
        prec1_aa = accuracy(output_aa.data, target)[0]

        top1.update(prec1.item(), input.size(0))
        top_fgsm.update(prec1_fgm.item(), input.size(0))
        top_pgd.update(prec1_pgd.item(), input.size(0))
        top_aa.update(prec1_aa.item(), input.size(0))

        loss = criterion(output, target_var)
        loss = loss.float()
        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        i, len(test_loader), batch_time=batch_time, loss=losses,
                        top1=top1))
    print(' * Prec@1 {:.3f} * Precfgsm@1 {:.3f} * Precpgd@1 {:.3f} * Precaa@1 {:.3f}'
      .format(top1.avg, top_fgsm.avg, top_pgd.avg, top_aa.avg))
    return ' * Prec@1 {:.3f} * Precfgsm@1 {:.3f} * Precpgd@1 {:.3f} * Precaa@1 {:.3f}'.format(top1.avg, top_fgsm.avg, top_pgd.avg, top_aa.avg)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    if is_best:
        torch.save(state, filename)

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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    # logger.critical(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
