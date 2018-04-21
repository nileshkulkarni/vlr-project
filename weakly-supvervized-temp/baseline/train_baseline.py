import torch
import torch.nn
import os.path as osp
from baseline.utils.parser import get_opts
from baseline.nnutils.stream_modules import ActionClassification
from baseline.data.ucf101 import UCF101, split
from baseline.logger import Logger
from tqdm import tqdm
import pdb
import collections


def train(epoch, model, optimizer, data_iter, logger):
  model.train()
  for batch_idx, batch in tqdm(enumerate(data_iter), desc='Epoch {} '.format(epoch), unit="iteration"):
    iteration = epoch * len(data_iter) + batch_idx
    outputs = model(batch['rgb'], batch['flow'])
    target_labels = batch['label']
    losses = model.build_loss(outputs, target_labels)
    
    total_loss = torch.mean(losses['total_loss'])
    
    info = dict([(key, torch.mean(item).data.cpu().numpy()[0]) for (key, item) in losses.items()])
    for tag, value in info.items():
      logger.scalar_summary(tag, value, iteration)
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()


def collate_fn(batch):
  collated_batch = dict()
  keys = batch[0].keys()
  for key in keys:
    collated_batch[key] = torch.stack([example[key] for example in batch])
  return collated_batch


def main(opts):
  video_names = split(opts.ucf_dir)
 
  dataset = UCF101(video_names, opts)
  data_iter = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                          shuffle=True, collate_fn=collate_fn)
  
  logger = Logger(osp.join(opts.cache_dir, 'logs'), 'baseline')

  
  action_net = ActionClassification(opts.feature_size, opts.num_classes)
  action_net.cuda()
  action_net_optimizer = torch.optim.SGD(action_net.parameters(),
                                         lr=opts.lr,
                                         momentum=opts.momentum)
  for epoch in range(opts.epochs):
    train(epoch, action_net, action_net_optimizer, data_iter, logger)
  
  return


if __name__ == "__main__":
  opts = get_opts()
  main(opts)
