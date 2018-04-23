import torch
import torch.nn
import os.path as osp
from baseline.utils.parser import get_opts
from baseline.nnutils.stream_modules import ActionClassification
from baseline.data.ucf101 import UCF101Temporal, split
from baseline.logger import Logger
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter
import collections
from sklearn.metrics import average_precision_score, recall_score
import numpy as np


def data_tsne_plot(data_iter):
  features = []
  labels = []
  writer = SummaryWriter('cachedir/tsne-log4/')
  
  for batch_idx, batch in enumerate(data_iter):
    rgb_features = torch.chunk(batch['rgb'], len(batch['rgb']),
                               0)
    target_labels = torch.chunk(batch['label'], len(batch['rgb']), 0)
    for feature, label in zip(rgb_features, target_labels):
      features.append(feature)
      labels.append(label[0].numpy()[0])

  labels = ['/{}/'.format(label) for label in labels]
  features = torch.cat(features, dim=0)
  print(features.size())
  writer.add_embedding(features, metadata=labels)
  writer.close()

def collate_fn(batch):
  collated_batch = dict()
  keys = batch[0].keys()
  for key in keys:
    key_features = []
    for example in batch:
      key_features.extend(example[key])
    collated_batch[key] = torch.stack(key_features)
  return collated_batch



from time import gmtime, strftime


def main(opts):
  video_names = split(opts.ucf_dir)
  
  dataset = UCF101Temporal('val', video_names, opts)
  
  data_iter = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size,
                                          shuffle=False, collate_fn=collate_fn)
  
  logdir = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
  
  
  
  data_tsne_plot(data_iter)
 

if __name__ == "__main__":
  opts = get_opts()
  main(opts)
