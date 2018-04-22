import torch
import os.path as osp
import os
from torch.utils.data import Dataset
## This claas loads the feature vector for the videos and the correspoding label.
import numpy as np
from torch.autograd import Variable
import pdb
import csv

class UCF101(Dataset):
  def __init__(self, dataset_name, video_names, opts):
    self._ucf_dir = opts.ucf_dir
    self._video_names = video_names
    self._feature_size = opts.feature_size
    
    self._file_names = []
    self._labels = []
    self.class_labels(dataset_name, opts.labels_dir)

    self._labels = []
    for file in os.listdir(self._ucf_dir):
      if file in self._video_names:
        self._file_names.append(file)
        video_index = self.video2index[file]
        self._labels.append(self.video_labels[video_index])
    
    self._labels = np.stack(self._labels)
    self._num_classes = opts.num_classes
    self._combine_startegy = opts.combine_strategy
    self._segments = opts.segments
    self._labels = torch.from_numpy(self._labels).float()
    # self._labels = torch.Tensor(len(self._file_names), self._num_classes).float().zero_()
    
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

  def class_labels(self, name, labels_dir):
    ##
   
    class2index_file =  osp.join(labels_dir, 'class_dict.csv')
    video2index_file = osp.join(labels_dir, 'video_indices_{}.csv'.format(name))
    video2labels_file = osp.join(labels_dir, 'class_labels_{}.npy'.format(name))
    
    self.class2index = dict()
    self.video2index = dict()
    self.video_labels = None
    with open(class2index_file,'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        self.class2index[row[0]] = int(row[1])

    with open(video2index_file,'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        self.video2index[row[0]] = int(row[1])
      
    self.video_labels = np.load(video2labels_file)
      
    

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


class UCF101Temporal(Dataset):
  def __init__(self, dataset_name, video_names, opts):
    self._ucf_dir = opts.ucf_dir
    self._video_names = video_names
    self._feature_size = opts.feature_size
    
    self._file_names = []
    self._labels = []
    self.class_labels(dataset_name, opts.labels_dir)
    
    self._labels = []
    self._video2segment_label = dict()
    for file in os.listdir(self._ucf_dir):
      if file in self._video_names:
        self._file_names.append(file)
        video_index = self.video2index[file]
        self._labels.append(self.video_labels[video_index])
    
    with open(os.path.join(opts.labels_dir, 'time_stamps.txt')) as f:
      for line in f:
        splits = line.strip().split(';')
        s = []
        for p in splits[1:]:
          s.append((int(float(p.split(',')[0])),int(float(p.split(',')[1])), int(p.split(',')[2])))

        self._video2segment_label[splits[0]] = s
    
    self._segment_positions_and_labels  = [] # (start, end, label)
    
    for f in self._file_names:
      self._segment_positions_and_labels.append(self._video2segment_label[f])
    

    self._labels = np.stack(self._labels)
    self._num_classes = opts.num_classes
    self._combine_startegy = opts.combine_strategy
    self._segments = opts.segments
    self._labels = torch.from_numpy(self._labels).float()
    
  def __len__(self):
    return len(self._file_names)
  
  def __getitem__(self, item):
    ## returns the feature vector for the video
    flow_features, rgb_features, labels = self.forward_video_as_segments(item)
    data = dict()
    data['flow'] = flow_features
    data['rgb'] = rgb_features
    data['label'] = labels
    return data
  
  def forward_video_as_segments(self, index):
   
    filename = self._file_names[index]
    flow_file = osp.join(self._ucf_dir, filename, 'features_flow.npy')
    rgb_file = osp.join(self._ucf_dir, filename, 'features_rgb.npy')
  
    flow_features = np.load(open(flow_file, 'r'))
    rgb_features = np.load(open(rgb_file, 'r'))
  
    flow_segments = []
    rgb_segments = []
    labels = []
    segments = self._segment_positions_and_labels[index]
    for s in segments:
      flow_segments.append(torch.from_numpy(np.mean(flow_features[s[0]:max(s[1],s[1]+1),:], axis=0)))
      rgb_segments.append(
          torch.from_numpy(np.mean(rgb_features[s[0]:max(s[1],s[1]+1), :], axis=0)))
      labels.append(torch.Tensor([s[2]]))
    return (flow_segments,rgb_segments, labels)
  
  
  def class_labels(self, name, labels_dir):
    class2index_file = osp.join(labels_dir, 'class_dict.csv')
    video2index_file = osp.join(labels_dir, 'video_indices_{}.csv'.format(name))
    video2labels_file = osp.join(labels_dir, 'class_labels_{}.npy'.format(name))
    
    self.class2index = dict()
    self.video2index = dict()
    self.video_labels = None
    with open(class2index_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        self.class2index[row[0]] = int(row[1])
    
    with open(video2index_file, 'r') as csvfile:
      reader = csv.reader(csvfile)
      for row in reader:
        self.video2index[row[0]] = int(row[1])
    
    self.video_labels = np.load(video2labels_file)
  
  def forward_label(self, index):
    return Variable(self._labels[index]).cuda()


def split(data_dir):
  all_files = []
  for file in os.listdir(data_dir):
    all_files.append(file)
  return all_files