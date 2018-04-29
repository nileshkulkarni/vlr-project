import torch
import torch.nn
import os.path as osp
from modular_attention.utils.parser import get_opts
from modular_attention.nnutils.stream_modules import ClassAttentionModule
from modular_attention.data.ucf101 import UCF101, split, UCF101_modular, \
  UCF101Temporal
from modular_attention.logger import Logger
from tqdm import tqdm
import pdb
from tensorboardX import SummaryWriter
import collections
from sklearn.metrics import average_precision_score, recall_score
import numpy as np
from torch.autograd import Variable


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
  p_mean_acc = AverageMeter()
  n_mean_acc = AverageMeter()
  
  with tqdm(enumerate(data_iter, 1), total=len(data_iter),
            desc='Epoch {} '.format(epoch), unit="iteration") as pbar:
    for batch_idx, batch in pbar:
      opts.log_now = 0
      iteration = epoch * len(data_iter) + batch_idx
      opts.iteration = iteration
      if logging == 1 and (iteration % opts.log_every) == 0:
        opts.log_now = 1
      
      ## assuming batch contains n frames, each with label, and class_id for batch
      
      # import pdb;pdb.set_trace()
      class_id = batch['pos_class'].data.cpu().numpy()[0][0]
      input_frames = batch['rgb_features']
      label_weights = batch['weights']
      # input_frames = batch[0]
      target_labels = batch['labels']
      
      output_labels = model.forward(input_frames,class_id)
      # import pdb;pdb.set_trace()
      loss = model.build_binary_loss(class_id, output_labels, target_labels,
                                     label_weights)
      acc = compute_accuracy(output_labels, target_labels)
      avg_loss.update(loss.data[0])
      class_wise_acc[class_id].update(acc[2].data[0])
      mean_acc.update(acc[2].data[0])
      p_mean_acc.update(acc[0].data[0])
      n_mean_acc.update(acc[1].data[0])
      
      pbar.set_postfix(loss=avg_loss.avg, mean_acc=mean_acc.avg,
                       p_mean_acc=p_mean_acc.avg)
      if logging == 1 and (iteration % opts.log_every) == 0:
        info = {'0/1_loss': loss.data[0], 'pos_mean_acc': p_mean_acc.avg,
                'neg_mean_acc': n_mean_acc.avg, 'mean_acc': mean_acc.avg,
                '{}'.format(class_id): acc[2].data[0]}
        for tag, value in info.items():
          logger.scalar_summary(tag, value, iteration)
      
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


def compute_accuracy(output_labels, target_labels):
  outputs = torch.ge(output_labels, 0.5).float()
  acc = (1 - torch.abs(outputs - target_labels)).mean()
  pacc = ((1 - torch.abs(outputs - target_labels)) * target_labels).sum() / (
        target_labels.sum() + 1E-5)
  nacc = ((1 - torch.abs(outputs - target_labels)) * (
        1 - target_labels)).sum() / ((1 - target_labels).sum() + 1E-5)
  return [pacc, nacc, acc]


def collate_fn(batch):
  collated_batch = dict()
  keys = batch[0].keys()
  for key in keys:
    collated_batch[key] = torch.stack([example[key] for example in batch])
  return collated_batch


def collate_fn_test(batch):
  collated_batch = torch.stack([example for example in batch])
  return collated_batch.view(1, -1, 1024)


def find_tps(target_labels, output_labels):
  output_labels = (output_labels > 0.5) * 1
  tp = np.sum(target_labels * output_labels)
  fp = np.sum((1 - target_labels) * output_labels)
  numPos = np.sum(target_labels)
  return tp, fp, numPos


def test(model, data_iter, opts):
  model.eval()
  tps = 0
  fps = 0
  numPos = 0
  for batch_idx, batch in enumerate(data_iter):
    opts.log_now = 0
    class_id = int(batch['label'][0][0][0])
    test_class_id = 0
    # import pdb;pdb.set_trace()
    input_frames = Variable(collate_fn_test(batch['rgb'])).cuda()
    output_labels = model.forward(input_frames, test_class_id).data.cpu().numpy()
    # import pdb;pdb.set_trace()
    target_labels = np.ones(output_labels.shape)
    if class_id != 0:
      target_labels = target_labels * 0
    tp, fp, numP = find_tps(target_labels, output_labels)
    tps = tps + tp
    fps = fps + fp
    numPos = numPos + numP
  
  print('True Positives: ', tps, ' False Positives : ', fps, ' Num Positives: ',
        numPos)


from time import gmtime, strftime


def main(opts):
  test_dataset = UCF101Temporal('val', opts)
  dataset = UCF101_modular('val', opts)
  
  test_data_iter = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                               shuffle=False)
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
                                              momentum=opts.momentum,
                                              weight_decay=opts.weight_decay)
  
  # data_tsne_plot(data_iter)
  
  for epoch in range(opts.epochs):
    if epoch % 4 == 0:
      test(class_attention_net, test_data_iter, opts)
    
    train(epoch, class_attention_net, class_attention_net_optim, data_iter,
          logger,
          opts)
  
  return


if __name__ == "__main__":
  opts = get_opts()
  main(opts)
