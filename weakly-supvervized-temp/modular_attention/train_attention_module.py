import torch
import torch.nn
import os.path as osp
from modular_attention.utils.parser import get_opts
from modular_attention.nnutils.stream_modules import ClassAttentionModule
from modular_attention.data.ucf101 import UCF101, split, UCF101_modular
from modular_attention.logger import Logger
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter
import collections
from sklearn.metrics import average_precision_score, recall_score
import numpy as np


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


logging = 1


def train(epoch, model, optimizer, data_iter, logger, opts):
  model.train()
  avg_loss = AverageMeter()
  class_wise_acc = []
  for i in range(opts.num_classes):
    class_wise_acc.append(AverageMeter())
  
  mean_acc = AverageMeter()
  
  with tqdm(enumerate(data_iter, 1), total=len(data_iter),
            desc='Epoch {} '.format(epoch), unit="iteration") as pbar:
    for batch_idx, batch in pbar:
      opts.log_now = 0
      iteration = epoch * len(data_iter) + batch_idx
      opts.iteration = iteration
      if logging == 1 and (iteration % opts.log_every) == 0:
        opts.log_now = 1
      
      ## assuming batch contains n frames, each with label, and class_id for batch
      
      #import pdb;pdb.set_trace()
      class_id = batch['pos_class']
      input_frames = batch['flow_features']
      #input_frames = batch[0]
      target_labels = batch['labels']
      
      output_labels = model(class_id, input_frames)
      loss = model.build_binary_loss(class_id, output_labels, target_labels)
      acc = compute_accuracy(output_labels, target_labels)
      avg_loss.update(loss.data[0])
      class_wise_acc[class_id].update(acc.data[0])
      mean_acc.update(acc.data[0])
      
      pbar.set_postfix(loss=avg_loss.avg, mean_acc = mean_acc.data[0])
      if logging == 1 and (iteration % opts.log_every) == 0:
        info = {'0/1_loss' : loss.data[0], 'mean_acc' :mean_acc.data[0],'{}'.format(class_id) : acc.data[0]}
        for tag, value in info.items():
          logger.scalar_summary(tag, value, iteration)
  
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


def compute_accuracy(output_labels, target_labels):
    outputs = torch.ge(output_labels, 0.5)
    acc = (1 - torch.abs(outputs - target_labels)).mean()
    return acc
    


def collate_fn(batch):
  collated_batch = dict()
  keys = batch[0].keys()
  for key in keys:
    collated_batch[key] = torch.stack([example[key] for example in batch])
  return collated_batch


from time import gmtime, strftime


def main(opts):
  video_names = split(opts.ucf_dir)
  
  dataset = UCF101_modular('val', opts)
  data_iter = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          shuffle=True, collate_fn=collate_fn)
  
  logdir = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
  
  opts.logger = logger = Logger(osp.join(opts.cache_dir, 'logs', logdir),
                                'baseline', )
  class_attention_net = ClassAttentionModule(opts.feature_size,
                                             opts.num_classes)
  class_attention_net.cuda()
  class_attention_net_optim = torch.optim.SGD(class_attention_net.parameters(),
                                         lr=opts.lr,
                                         momentum=opts.momentum)
  
  # data_tsne_plot(data_iter)
  # pdb.set_trace()
  for epoch in range(opts.epochs):
    train(epoch, class_attention_net, class_attention_net_optim, data_iter, logger,
          opts)
  return


if __name__ == "__main__":
  opts = get_opts()
  main(opts)
