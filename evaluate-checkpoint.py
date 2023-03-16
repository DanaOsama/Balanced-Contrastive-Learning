import torch
import time
import shutil
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from loss.contrastive import BalSCL
from loss.logitadjust import LogitAdjust, FocalLoss, FocalLC
import math
from tensorboardX import SummaryWriter
from dataset.mydataset import MyDataset
# from dataset.imagenet import ImageNetLT
from models import resnext
import warnings
import torch.backends.cudnn as cudnn
import random
from randaugment import rand_augment_transform
import torchvision
from utils import GaussianBlur, shot_acc, get_random_string
# from torch.models.tensorboard import SummaryWriter
import argparse
import random
import string
import os
from sklearn.metrics import f1_score
from tqdm import tqdm
import time
import wandb
from datetime import datetime


checkpoint_path = "log/isic_resnet50_batchsize_128_epochs_1000_temp_0.07_lr_0.05_sim-sim_alpha_1.0_beta_0.35_schedule_[860, 880]_recalibrate_False_Salwa_ce_loss_LC/bcl_ckpt.best.pth.tar"

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

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calc_f1(output, target):
    """
    It takes the output of the model and the target, and returns the F1 score
    
    :param output: the output of the model, which is a tensor of shape (batch_size, num_classes)
    :param target: the ground truth labels
    """
    with torch.no_grad():
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()

    return [f1_score(target.cpu(), pred.squeeze(0).cpu(), average='macro')]

def validate(train_loader, val_loader, model, criterion_ce, epoch, tf_writer=None, flag='val'):
    model.eval()
    batch_time = AverageMeter('Time', ':6.3f')
    ce_loss_all = AverageMeter('CE_Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    f1 = AverageMeter('F1', ':.4e')
    total_logits = torch.empty((0, 7)).cuda()
    total_labels = torch.empty(0, dtype=torch.long).cuda()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(tqdm(val_loader)):
            inputs, targets = data
            inputs, targets = inputs.cuda(), targets.cuda()
            batch_size = targets.size(0)
            feat_mlp, logits, centers = model(inputs, phase = 'val')
            ce_loss = criterion_ce(logits, targets)

            total_logits = torch.cat((total_logits, logits))
            total_labels = torch.cat((total_labels, targets))

            acc1 = accuracy(logits, targets, topk=(1,))
            f1_acc = calc_f1(logits, targets)
            ce_loss_all.update(ce_loss.item(), batch_size)
            top1.update(acc1[0].item(), batch_size)
            f1.update(f1_acc[0].item(), batch_size)

            batch_time.update(time.time() - end)

        # if i % args.print_freq == 0:
        output = ('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'CE_Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                'F1 score {f1.val:.4f} ({f1.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
            i, len(val_loader), batch_time=batch_time, ce_loss=ce_loss_all,f1=f1, top1=top1, )) 
        
        # wandb.log({"ce_loss_val_avg": ce_loss_all.avg, "top1_val_avg": top1.avg, "f1_val_avg": f1.avg}, step=epoch) 

        print(output)

        # tf_writer.add_scalar('CE loss/val', ce_loss_all.avg, epoch)
        # tf_writer.add_scalar('acc/val_top1', top1.avg, epoch)

        # probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)
        # many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds, total_labels, train_loader, many_shot_thr=args.many_shot_thr, low_shot_thr=args.low_shot_thr,
        #                                                         acc_per_cls=False)
        return top1.avg, f1.avg
    

val_data = '/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input'
txt_val = f'/l/users/salwa.khatib/proco/ISIC2018_Task3_Validation_Input/ISIC2018_Task3_Validation_GroundTruth.txt'

normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


val_dataset = MyDataset(
    root=val_data,
    txt=txt_val,
    transform=val_transform, train=False, num_classes = 7
)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=110, shuffle=False,
    num_workers=32, pin_memory=True)

model = resnext.BCLModel(name='resnet50', num_classes=7, feat_dim=1024,
                        use_norm=True, recalibrate = False)

model = torch.nn.DataParallel(model).cuda()

ckpt = torch.load(checkpoint_path)['state_dict']
# print(ckpt.keys())
model.load_state_dict(ckpt)
print("[INFO] loaded checkpoint")

cls_num_list = val_dataset.cls_num_list
criterion_ce = LogitAdjust(cls_num_list)

validate(val_loader, val_loader, model, criterion_ce, 1, tf_writer=None, flag='val')