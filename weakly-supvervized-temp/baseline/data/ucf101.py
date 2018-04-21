import torch
import os.path as osp
import os
from torch.utils.data import Dataset
## This claas loads the feature vector for the videos and the correspoding label.
import numpy as np
from torch.autograd import Variable
import pdb

class UCF101(Dataset):
  def __init__(self, video_names, opts):
    self._ucf_dir = opts.ucf_dir
    self._video_names = video_names
    self._feature_size = opts.feature_size
    
    self._file_names = []
    self._labels = []
    for file in os.listdir(self._ucf_dir):
      if file in self._video_names:
        self._file_names.append(file)
    self._num_classes = opts.num_classes
    self._combine_startegy = opts.combine_strategy
    self._segments = opts.segments
    self._labels = torch.Tensor(len(self._file_names), self._num_classes).float().zero_()
  def __len__(self):
    return len(self._file_names)
  
  def __getitem__(self, item):
    ## returns the feature vector for the video
    flow_features, rgb_features = self.forward_video(item)
    label = self.forward_label(item)
    data = dict()
    data['flow'] = flow_features
    data['rgb'] = rgb_features
    data['label'] = label
    return data

  def forward_label(self, index):
    return Variable(self._labels[index]).cuda()
  
  def forward_video(self, index):
    filename = self._file_names[index]
    flow_file = osp.join(self._ucf_dir, filename,'features_flow.npy')
    rgb_file = osp.join(self._ucf_dir, filename, 'features_rgb.npy')
    
    
    flow_features = np.load(open(flow_file,'r'))
    rgb_features = np.load(open(rgb_file, 'r'))
    
    flow_segments = np.zeros((self._segments, self._feature_size), dtype=np.float32)
    rgb_segments = np.zeros((self._segments, self._feature_size),
                             dtype=np.float32)

    frames = flow_segments.shape[0]
    segment_len = flow_features.shape[0] // self._segments  + 1
    
    
    if self._combine_startegy == 'uniform':
      for i in range(self._segments):
        start = (i*segment_len)
        end = (i+1)*segment_len
        
        flow_segments[i, :] = np.mean(np.take(flow_features,np.arange(start, end), axis=0, mode='wrap'), axis=0)
        rgb_segments[i, :] = np.mean(np.take(rgb_features,np.arange(start, end), axis=0, mode='wrap'), axis=0)

    ## rgb_feautes are of the T depending on the length of the video.
    #  Each segment has 1024 dimensional feature.
    flow_segments = Variable(torch.from_numpy(flow_segments).cuda())
    rgb_segments = Variable(torch.from_numpy(rgb_segments).cuda())

    return flow_segments, rgb_segments
  

    

def split(data_dir):
  all_files = []
  for file in os.listdir(data_dir):
    all_files.append(file)
  return all_files