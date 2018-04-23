import torch
import torch.nn
import os.path as osp
from baseline.utils.parser import get_opts
from baseline.nnutils.stream_modules import ActionClassification
from baseline.data.ucf101 import UCF101, split
from baseline.logger import Logger
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


def data_tsne_plot(data_iter):
  features = []
  labels = []
  writer = SummaryWriter('cachedir/tsne-log')
  for batch_idx, batch in enumerate(data_iter):
    rgb_features = torch.chunk(torch.mean(batch['rgb'], 1), len(batch['rgb']),
                               0)
    target_labels = torch.chunk(batch['label'], len(batch['rgb']), 0)
    for feature, label in zip(rgb_features, target_labels):
      features.extend(feature.data)
      # label_index = label[]
      # labels.extend()
  
  features = torch.stack(features)
  writer.add_embedding(features)
  writer.close()


logging = 1


def train(epoch, model, optimizer, data_iter, logger, opts):
  model.train()
  avg_loss = AverageMeter()
  avg_precision = AverageMeter()
  with tqdm(enumerate(data_iter, 1), total=len(data_iter),
            desc='Epoch {} '.format(epoch), unit="iteration") as pbar:
    for batch_idx, batch in pbar:
      iteration = epoch * len(data_iter) + batch_idx
      outputs = model(batch['rgb'], batch['flow'])
      target_labels = batch['label']
      losses = model.build_loss(outputs, target_labels)
      recall, precision = compute_accuracy(
        outputs['class_rgb'].data.cpu().numpy(),
        target_labels.data.cpu().numpy())
      total_loss = torch.mean(losses['total_loss'])
      # print(total_loss)
      avg_loss.update(total_loss.data[0])
      avg_precision.update(precision)
      pbar.set_postfix(loss=avg_loss.avg, precision=avg_precision.avg)
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
      
      optimizer.zero_grad()
      total_loss.backward()
      optimizer.step()


def compute_accuracy(output_labels, target_labels):
  output_labels = 1 * (output_labels > 0.5)
  recall = 0
  target_labels = target_labels.astype(np.int32)
  precision = average_precision_score(target_labels, output_labels,
                                      average='micro')
  return recall, precision


def collate_fn(batch):
  collated_batch = dict()
  keys = batch[0].keys()
  for key in keys:
    collated_batch[key] = torch.stack([example[key] for example in batch])
  return collated_batch


from time import gmtime, strftime


def main(opts):
  video_names = split(opts.ucf_dir)
  
  dataset = UCF101('val', video_names, opts)
  data_iter = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                          shuffle=True, collate_fn=collate_fn)
  
  logdir = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
  
  logger = Logger(osp.join(opts.cache_dir, 'logs', logdir), 'baseline', )
  
  action_net = ActionClassification(opts.feature_size, opts.num_classes, opts)
  action_net.cuda()
  action_net_optimizer = torch.optim.SGD(action_net.parameters(),
                                         lr=opts.lr,
                                         momentum=opts.momentum)
  
  # data_tsne_plot(data_iter)
  # pdb.set_trace()
  for epoch in range(opts.epochs):
    train(epoch, action_net, action_net_optimizer, data_iter, logger, opts)
  return


if __name__ == "__main__":
  opts = get_opts()
  main(opts)
