import torch
import os.path as osp
import os
from torch.utils.data import Dataset
## This claas loads the feature vector for the videos and the correspoding label.
import numpy as np
from torch.autograd import Variable
import pdb
import csv
import collections


class UCF101(Dataset):
  def __init__(self, dataset_name, opts):
    self._ucf_dir = osp.join(opts.ucf_dir, "{}_features".format(dataset_name))
    self._ignore_names = [".", ".."]
    self._feature_size = opts.feature_size
    
    self._file_names = []
    self._labels = []
    self.class_labels(dataset_name, opts.labels_dir)
    
    self._labels = []
    for file in os.listdir(self._ucf_dir):
      if file not in self._ignore_names:
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
    flow_features, rgb_features, numInputs = self.forward_video(item)
    label = self.forward_label(item)
    data = dict()
    data['flow'] = flow_features
    data['rgb'] = rgb_features
    data['label'] = label
    data['numInputs'] = numInputs
    return data
  
  def class_labels(self, name, labels_dir):
    ##
    
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
  
  def forward_video(self, index):
    filename = self._file_names[index]
    flow_file = osp.join(self._ucf_dir, filename, 'features_flow.npy')
    rgb_file = osp.join(self._ucf_dir, filename, 'features_rgb.npy')
    
    flow_features = np.load(open(flow_file, 'r'))
    rgb_features = np.load(open(rgb_file, 'r'))
    
    flow_segments = np.zeros((self._segments, self._feature_size),
                             dtype=np.float32)
    rgb_segments = np.zeros((self._segments, self._feature_size),
                            dtype=np.float32)
    
    frames = flow_segments.shape[0]
    segment_len = flow_features.shape[0] // self._segments + 1
    total_segments = flow_features.shape[0]
    numInputs = np.expand_dims(np.array([flow_features.shape[0]]), axis=0)
    
    if self._combine_startegy == 'uniform':
      for i in range(self._segments):
        start = (i * segment_len)
        end = (i + 1) * segment_len
        seq = np.arange(start, end)
        flow_segments[i, :] = np.mean(
          np.take(flow_features, seq, axis=0, mode='wrap'), axis=0)
        rgb_segments[i, :] = np.mean(
          np.take(rgb_features, seq, axis=0, mode='wrap'), axis=0)
    
    if self._combine_startegy == 'strat1':
      offset = np.random.choice(segment_len, 1)
      for i in range(self._segments):
        start = (i * segment_len) + offset
        end = (i + 1) * segment_len + offset
        seq = np.arange(start, end)
        flow_segments[i, :] = np.mean(
          np.take(flow_features, seq, axis=0, mode='wrap'),
          axis=0)
        rgb_segments[i, :] = np.mean(
          np.take(rgb_features, seq, axis=0, mode='wrap'),
          axis=0)
    
    if self._combine_startegy == 'strat2':
      for i in range(self._segments):
        start = (i * segment_len)
        end = (i + 1) * segment_len
        if segment_len > 1:
          seq = np.random.choice(segment_len, int(segment_len * 0.8)) + start
        else:
          seq = np.arange(start, end)
        flow_segments[i, :] = np.mean(
          np.take(flow_features, seq, axis=0, mode='wrap'),
          axis=0)
        rgb_segments[i, :] = np.mean(
          np.take(rgb_features, seq, axis=0, mode='wrap'),
          axis=0)
    if self._combine_startegy == 'strat3':
      for i in range(self._segments):
        sample_range = total_segments
        while sample_range < self._segments + 1:
          sample_range = sample_range + total_segments
        
        sampledN = np.round(
          np.linspace(0, sample_range, self._segments + 1)).astype(np.int32)
        # import pdb;pdb.set_trace()
        
        differences = sampledN[1:] - sampledN[0:-1]
        randoms = np.random.rand(self._segments)
        K = sampledN[0:-1] + np.round(randoms * differences).astype(np.int)
        K = np.mod(K, np.ones(K.shape) * total_segments).astype(np.int)
        flow_segments = flow_features[K, :]
        rgb_segments = rgb_features[K, :]
    
    ## rgb_feautes are of the T depending on the length of the video.
    #  Each segment has 1024 dimensional feature.
    flow_segments = Variable(torch.from_numpy(flow_segments).cuda())
    rgb_segments = Variable(torch.from_numpy(rgb_segments).cuda())
    numInputs = Variable(torch.from_numpy(numInputs))
    
    return flow_segments, rgb_segments, numInputs


class UCF101Temporal(Dataset):
    def __init__(self, dataset_name, video_names, opts):
        self._ucf_dir = opts.ucf_dir
        self._ucf_dir = '/scratch/smynepal/THUMOSFrames/val_features/'
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
                    s.append((int(float(p.split(',')[0])), int(float(p.split(',')[1])), int(p.split(',')[2])))

                self._video2segment_label[splits[0]] = s

        self._segment_positions_and_labels = []  # (start, end, label)

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
        flow_features, rgb_features, labels, filename = self.forward_video_as_segments(item)
        data = dict()
        data['flow'] = flow_features
        data['rgb'] = rgb_features
        data['label'] = labels
        #data['video_name'] = filename
        #import pdb;pdb.set_trace()
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
            if s[0] < rgb_features.shape[0]:
                flow_segments.append(torch.from_numpy(np.mean(flow_features[s[0]:max(s[1], s[1] + 1), :], axis=0)))
                rgb_segments.append(
                    torch.from_numpy(np.mean(rgb_features[s[0]:max(s[1], s[1] + 1), :], axis=0)))
                labels.append(torch.Tensor([s[2]]))
        return (flow_segments, rgb_segments, labels, filename)

    def class_labels(self, name, labels_dir):
        labels_dir = '/home/smynepal/Projects/VLR/I3D/train/vlr-project/weakly-supvervized-temp/thumos_data/labels/'
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


# opts should additionally have the following attrinutes
# 1. number of frames of same class, maybe 800
# 2. number of frames of different class, may be 800
# need to be sufficiently large because we want suffient number of frames to actually contain the activity
# 3. Penalize negative examples being classified as positive more? To separate the non-representative frames further
# Assuming that classification and 0/1 binary weight prediction are trained iteratively


class UCF101_modular(Dataset):
    def __init__(self, dataset_name, opts):
        self._ucf_dir = osp.join(opts.ucf_dir, "{}_features".format(dataset_name))
        self._ignore_names = [".", ".."]
        self._feature_size = opts.feature_size

        self._file_names = []
        self._labels = []
        self.class_labels(dataset_name, opts.labels_dir)

        self._labels = []
        for file in os.listdir(self._ucf_dir):
            if file not in self._ignore_names:
                self._file_names.append(file)
                video_index = self.video2index[file]
                self._labels.append(self.video_labels[video_index])
        self._labels = np.stack(self._labels)
        self._num_classes = opts.num_classes
        self._combine_startegy = opts.combine_strategy
        self._segments = opts.segments
        self._labels = torch.from_numpy(self._labels).float()
        # self._labels = torch.Tensor(len(self._file_names), self._num_classes).float().zero_()

        # parameters for independent classifer training data,
        self.weight_pos = opts.weight_pos
        self.weight_neg = opts.weight_neg
        self._num_same = opts.num_same
        self._num_diff = opts.num_diff
        self._num_class_iter_per_epoch = opts.num_class_iter_per_epoch
        self.label_index_dict = self.create_label_index_dict()
        self.flow_features_all, self.rgb_features_all = self.read_all_features()

    def __len__(self):
        return self._num_class_iter_per_epoch * self._num_classes  # 1 peoch is when, the network sees a class approximately N number of times

    def __getitem__(self, item):
        ## returns the feature vector for the video
        # import pdb;pdb.set_trace()
        pos_class, labels, weights, flow_features, rgb_features = self.forward_video(item)
        data = dict()
        data['weights'] = weights
        data['labels'] = labels
        data['flow_features'] = flow_features
        data['rgb_features'] = rgb_features
        data['pos_class'] = pos_class
        return data

    def class_labels(self, name, labels_dir):
        ##

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

    def create_label_index_dict(self):
        label_index_dict_pos = collections.defaultdict(list)
        label_index_dict = dict()
        for idx in range(len(self.video_labels)):
            label = self.video_labels[idx]
            cls_inds = np.where(label == 1)[0]
            for j in range(len(cls_inds)):
                label_index_dict_pos[cls_inds[j]].append(idx)

        for j in range(self._num_classes):
            pos_label_inds = label_index_dict_pos[j]
            neg_label_inds = [x for x in range(len(self.video_labels)) if x not in pos_label_inds]
            label_index_dict[j] = tuple([pos_label_inds, neg_label_inds])
        return label_index_dict

    def read_all_features(self):
        flow_features_all = []
        rgb_features_all = []
        for filename in self._file_names:
            flow_file = osp.join(self._ucf_dir, filename, 'features_flow.npy')
            rgb_file = osp.join(self._ucf_dir, filename, 'features_rgb.npy')

            flow_features = np.load(open(flow_file, 'r'))
            rgb_features = np.load(open(rgb_file, 'r'))

            flow_features_all.append(flow_features)
            rgb_features_all.append(rgb_features)

        return flow_features_all, rgb_features_all

    def forward_video(self, index):
        # 1. randomly generate class.
        #np.random.seed(0)
        #index = 0
		
        cls_index = np.random.choice(self._num_classes, 1)[0]
        cls_index = 0
        pos_vid_inds, neg_vid_inds = self.label_index_dict[cls_index]

        # choosing with replacement 8 times
        sample_pos_inds = np.random.choice(pos_vid_inds, 8)
        sample_neg_inds = np.random.choice(neg_vid_inds, 12)

        # sampling pos features flow
        pos_features_flow = np.vstack([self.flow_features_all[i] for i in sample_pos_inds])
        # sampling 800 from them
        if len(pos_features_flow) > self._num_same:
            flow_input_pos = pos_features_flow[np.random.choice(len(pos_features_flow), self._num_same, replace=False),
                             :]
        else:
            flow_input_pos = pos_features_flow[np.random.choice(len(pos_features_flow), self._num_same), :]

        # clearing memory pos flow
        del pos_features_flow

        # sampling pos features rgb
        pos_features_rgb = np.vstack([self.rgb_features_all[i] for i in sample_pos_inds])
        # sampling 800 from them
        if len(pos_features_rgb) > self._num_same:
            rgb_input_pos = pos_features_rgb[np.random.choice(len(pos_features_rgb), self._num_same, replace=False), :]
        else:
            rgb_input_pos = pos_features_rgb[np.random.choice(len(pos_features_rgb), self._num_same), :]

        # clearing memory pos flow
        del pos_features_rgb

        # sampling neg features flow
        neg_features_flow = np.vstack([self.flow_features_all[i] for i in sample_neg_inds])
        # sampling 800 from them
        if len(neg_features_flow) > self._num_diff:
            flow_input_neg = neg_features_flow[
                             np.random.choice(len(neg_features_flow), self._num_diff, replace=False), :]
        else:
            flow_input_neg = neg_features_flow[np.random.choice(len(neg_features_flow), self._num_diff), :]

        # clearing memory neg flow
        del neg_features_flow

        # sampling neg features rgb
        neg_features_rgb = np.vstack([self.rgb_features_all[i] for i in sample_neg_inds])
        # sampling 800 from them
        if len(neg_features_rgb) > self._num_diff:
            rgb_input_neg = neg_features_rgb[
                            np.random.choice(len(neg_features_rgb), self._num_diff, replace=False), :]
        else:
            rgb_input_neg = neg_features_rgb[np.random.choice(len(neg_features_rgb), self._num_diff), :]

        # clearing memory neg rgb
        del neg_features_rgb

        labels_pos = np.ones([len(flow_input_pos),1])
        labels_neg = np.zeros([len(flow_input_neg),1])
        labels = np.reshape(np.vstack([labels_pos, labels_neg]), -1)
        labels = np.expand_dims(np.array(labels), axis=0)
        weights = labels * self.weight_pos + (1 - labels) * self.weight_neg
        #labels = np.expand_dims(np.array(labels), axis=0)


        flow_features_input = np.vstack([flow_input_pos, flow_input_neg])
        rgb_features_input = np.vstack([rgb_input_pos, rgb_input_neg])

        indices = [x for x in range(len(rgb_features_input))]
        np.random.shuffle(indices)

        #import pdb;pdb.set_trace()
        flow_features_input = Variable(torch.from_numpy(flow_features_input[indices]).cuda())
        rgb_features_input = Variable(torch.from_numpy(rgb_features_input[indices]).cuda())
        labels = Variable(torch.from_numpy(labels[0,indices]).float().cuda())
        weights = Variable(torch.from_numpy(weights[0,indices]).float().cuda())

        cls_index = np.expand_dims(np.array(cls_index), axis=0)
        cls_index = Variable(torch.from_numpy(cls_index))

        return cls_index, labels, weights, flow_features_input, rgb_features_input


def split(data_dir):
  all_files = []
  for file in os.listdir(data_dir):
    all_files.append(file)
  return all_files
