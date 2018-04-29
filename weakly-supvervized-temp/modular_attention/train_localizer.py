import torch
import torch.nn
import os.path as osp
from modular_attention.utils.parser import get_opts
from modular_attention.nnutils.stream_modules import ClassAttentionModule, \
  ActionClassification
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


def valid(epoch, model, data_iter, logger, opts):
  model.eval()
  avg_loss = AverageMeter()
  avg_precision = AverageMeter()
  with tqdm(enumerate(data_iter, 1), total=len(data_iter),
            desc='VL Epoch {} '.format(epoch), unit="iteration") as pbar:
    for batch_idx, batch in pbar:
      opts.log_now = 0
      outputs = model(batch['rgb'], batch['flow'])
      target_labels = batch['label']
      losses = model.build_loss(outputs, target_labels)
      recall, precision = compute_precision_recall(
        outputs['class_rgb'].data.cpu().numpy(),
        target_labels.data.cpu().numpy())
      total_loss = torch.mean(losses['total_loss'])
      avg_loss.update(total_loss.data[0])
      avg_precision.update(precision)
      pbar.set_postfix(loss=avg_loss.avg, precision=avg_precision.avg)
    info = {'valid/precision': avg_precision.avg}
    for key, value in info.items():
      logger.scalar_summary(key, value, opts.iteration)
  
  return avg_precision.avg


def train(epoch, action_model, action_net_optim,
          class_attention_model, class_attention_optim,
          data_iter_localization, data_iter_modular,
          logger, opts):
  action_model.train()
  avg_loss = AverageMeter()
  avg_precision = AverageMeter()
  
  avg_loss_modular = AverageMeter()
  class_wise_acc = []
  for i in range(opts.num_classes):
    class_wise_acc.append(AverageMeter())
  
  mean_acc = AverageMeter()
  p_mean_acc = AverageMeter()
  n_mean_acc = AverageMeter()
  
  iterator_modular = iter(data_iter_modular)
  
  with tqdm(enumerate(data_iter_localization, 1),
            total=len(data_iter_localization),
            desc='TR Epoch {} '.format(epoch), unit="iteration") as pbar:
    for batch_idx, batch in pbar:
      ## optimize the attention module
      opts.log_now = 0
      iteration = epoch * len(data_iter_localization) + batch_idx
      opts.iteration = iteration
      if logging == 1 and (iteration % opts.log_every) == 0:
        opts.log_now = 1
      
      outputs = action_model(batch['rgb'], batch['flow'])
      target_labels = batch['label']
      losses = action_model.build_loss(outputs, target_labels)
      recall, precision = compute_precision_recall(
        outputs['class_rgb'].data.cpu().numpy(),
        target_labels.data.cpu().numpy())
      total_loss = torch.mean(losses['total_loss'])
      # print(total_loss)
      avg_loss.update(total_loss.data[0])
      avg_precision.update(precision)
      
      if logging == 1 and (iteration % opts.log_every) == 0:
        info = dict(
          [(key, torch.mean(item).data.cpu().numpy()[0]) for (key, item) in
           losses.items()])
        for tag, value in info.items():
          logger.scalar_summary(tag, value, iteration)
        
        logger.histo_summary('attn_rgb',
                             outputs['attn_rgb'].data.cpu().numpy(),
                             iteration)
        logger.histo_summary('activation_rgb',
                             outputs['class_rgb'].data.cpu().numpy(),
                             iteration)
      
      action_net_optim.zero_grad()
      total_loss.backward()
      action_net_optim.step()
      
      ## Train Class specific Attention modules
      modular_batch = next(iterator_modular, None)
      
      if modular_batch is None:
        iterator_modular = iter(data_iter_modular)
        modular_batch = next(iterator_modular, None)
      
      class_id = modular_batch['pos_class'].data.cpu().numpy()[0][0]
      input_frames = modular_batch['rgb_features']
      label_weights = modular_batch['weights']
      # input_frames = batch[0]
      target_labels = modular_batch['labels']
      
      output_labels = class_attention_model(input_frames, class_id)
      # import pdb;pdb.set_trace()
      loss = class_attention_model.build_binary_loss(class_id, output_labels,
                                                     target_labels,
                                                     label_weights)
      acc = compute_accuracy(output_labels, target_labels)
      avg_loss_modular.update(loss.data[0])
      class_wise_acc[class_id].update(acc[2].data[0])
      mean_acc.update(acc[2].data[0])
      p_mean_acc.update(acc[0].data[0])
      n_mean_acc.update(acc[1].data[0])
      
      pbar.set_postfix(loss_modular=avg_loss.avg, mean_acc=mean_acc.avg,
                       p_mean_acc=p_mean_acc.avg, loss=avg_loss.avg,
                       precision=avg_precision.avg)
      
      if logging == 1 and (iteration % opts.log_every) == 0:
        info = {'0/1_loss': loss.data[0], 'pos_mean_acc': p_mean_acc.avg,
                'neg_mean_acc': n_mean_acc.avg, 'mean_acc': mean_acc.avg,
                '{}'.format(class_id): acc[2].data[0]}
        for tag, value in info.items():
          logger.scalar_summary(tag, value, iteration)
      
      class_attention_optim.zero_grad()
      loss.backward()
      class_attention_optim.step()


def compute_accuracy(output_labels, target_labels):
  outputs = torch.ge(output_labels, 0.5).float()
  acc = (1 - torch.abs(outputs - target_labels)).mean()
  pacc = ((1 - torch.abs(outputs - target_labels)) * target_labels).sum() / (
      target_labels.sum() + 1E-5)
  nacc = ((1 - torch.abs(outputs - target_labels)) * (
      1 - target_labels)).sum() / ((1 - target_labels).sum() + 1E-5)
  return [pacc, nacc, acc]


def compute_precision_recall(output_labels, target_labels):
  output_labels = 1 * (output_labels > 0.5)
  recall = 0
  target_labels = target_labels.astype(np.int32)
  precision = average_precision_score(target_labels, output_labels,
                                      average=None)
  precision = np.nanmean(precision)
  return recall, precision


def save_model(model, name, best):
  torch.save({'state_dict': model.state_dict()}, '{}.pkl'.format(name))
  if best:
    torch.save({'state_dict': model.state_dict()}, '{}_best.pkl'.format(name))


def load_model(model, name, best):
  checkpoint = torch.load('{}.pkl'.format(name))
  if best:
    checkpoint = torch.load('{}_best.pkl'.format(name))
  model.load_state_dict(checkpoint['state_dict'])


def collate_fn(batch):
  collated_batch = dict()
  keys = batch[0].keys()
  for key in keys:
    collated_batch[key] = torch.stack([example[key] for example in batch])
  return collated_batch


from time import gmtime, strftime


def main(opts):
  dataset_modular_train = UCF101_modular('val', opts)
  
  data_iter_modular = torch.utils.data.DataLoader(dataset_modular_train,
                                                  batch_size=1,
                                                  shuffle=True,
                                                  collate_fn=collate_fn)
  
  dataset_localization_valid = UCF101('test', opts)
  dataset_localization_train = UCF101('val', opts)
  
  data_iter_localization_train = torch.utils.data.DataLoader(
    dataset_localization_train,
    batch_size=opts.batch_size,
    shuffle=True,
    collate_fn=collate_fn)
  data_iter_localization_valid = torch.utils.data.DataLoader(
    dataset_localization_valid,
    batch_size=opts.batch_size,
    shuffle=True,
    collate_fn=collate_fn)
  
  logdir = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
  
  opts.logger = logger = Logger(osp.join(opts.cache_dir, 'logs', logdir),
                                'modular_attention', )
  
  action_net = ActionClassification(opts.feature_size, opts.num_classes, opts)
  class_attention_net_rgb = action_net.get_attention_models()[
    'rgb_attention_model']
  class_attention_net_optim = torch.optim.SGD(
    class_attention_net_rgb.parameters(),
    lr=1E-3,
    momentum=opts.momentum,
    weight_decay=1E-5)
  # action_net_optim = torch.optim.SGD(action_net.parameters(), lr=opts.lr,
  #                                    momentum=opts.momentum)
  action_net_optim = torch.optim.Adam(action_net.parameters(), lr=opts.lr)
  action_net.cuda()
  best_valid_precision = 0
  for epoch in range(1, opts.epochs):
    train(epoch, action_net, action_net_optim, class_attention_net_rgb,
          class_attention_net_optim,
          data_iter_localization_train, data_iter_modular,
          logger,
          opts)
    if epoch % 4 == 0 and epoch > 0:
      valid_precision = valid(epoch, action_net, data_iter_localization_valid,
                              logger, opts)
      best = valid_precision > best_valid_precision
      save_model(action_net,
                 osp.join(opts.cache_dir, 'models', 'action_localizer'), best)
  return


if __name__ == "__main__":
  opts = get_opts()
  main(opts)
