import torch
import torch.nn as nn


class ActionClassification(nn.Module):
  def __init__(self, feature_size, num_classes, opts):
    super(ActionClassification, self).__init__()
    self.opts = opts
    self.attention_flow_stream = StreamModule(feature_size)
    self.attention_rgb_stream = StreamModule(feature_size)
    self.classifier_flow_stream = StreamClassificationHead(feature_size, num_classes)
    self.classifier_rgb_stream = StreamClassificationHead(feature_size, num_classes)
    self.classifier_both = StreamClassificationHead(feature_size, num_classes)
    self.multi_label_cross = nn.MultiLabelSoftMarginLoss()
    

  def forward(self, rgb_features, flow_features):
    wt_ft_rgb, attn_rgb = self.attention_rgb_stream(rgb_features)
    wt_ft_flow, attn_flow = self.attention_flow_stream(flow_features)
    class_rgb = self.classifier_flow_stream(wt_ft_flow)
    class_flow = self.classifier_rgb_stream(wt_ft_rgb)
    features_both = wt_ft_flow + wt_ft_rgb
    class_both = self.classifier_both(features_both)
    outputs = dict()
    outputs['class_both'] = class_both
    outputs['class_rgb'] = class_rgb
    outputs['class_flow'] = class_flow
    outputs['attn_rgb'] = attn_rgb
    outputs['attn_flow'] = attn_flow
   
    return outputs
  
  def build_loss(self, outputs, target_class):
    # target_class is B x C
    self.rgb_loss = self.multi_label_cross(outputs['class_rgb'], target_class)
    self.flow_loss = self.multi_label_cross(outputs['class_flow'], target_class)
    self.both_loss = self.multi_label_cross(outputs['class_both'], target_class)
    self.rgb_sparsity = self.attention_rgb_stream.attention_module.l1_sparsity_loss(
      outputs['attn_rgb'])
    self.flow_sparsity = self.attention_flow_stream.attention_module.l1_sparsity_loss(
      outputs['attn_flow'])
    losses = {}
    losses['rgb_loss'] = self.opts.cls_wt*self.rgb_loss
    losses['flow_loss'] = self.opts.cls_wt*self.flow_loss
    losses['both_loss'] = self.opts.cls_wt*self.both_loss
    losses['rbg_sparsity_loss'] = self.opts.attn_wt*self.rgb_sparsity
    losses['flow_sparsity_loss'] = self.opts.attn_wt*self.flow_sparsity
    
    self.total_loss = self.rgb_loss + self.flow_loss + self.both_loss \
                      + self.rgb_sparsity + self.flow_sparsity
    losses['total_loss'] = self.total_loss
    return losses
class StreamModule(nn.Module):
  def __init__(self, feature_size):
    super(StreamModule, self).__init__()
    self.feature_size = feature_size
    self.attention_module = AttentionModule(self.feature_size)
  
  def forward(self, x):
    # x is B x T x 1024
    attention = self.attention_module(x)  # B x T
    attention_expand = attention.expand(x.size())
    new_features = x + x * attention_expand*0
    weighted_features = torch.sum(new_features, 1)
    return weighted_features, attention


class StreamClassificationHead(nn.Module):
  def __init__(self, feature_size, num_classes):
    super(StreamClassificationHead, self).__init__()
    self.classifier = nn.Linear(feature_size, num_classes)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, x):
    x = self.classifier(x)
    x = self.sigmoid(x)
    return x


## It becomes easier if we have a fixed number of segments?
## Batch processing is faster.
## Rethink about it?

class AttentionModule(nn.Module):
  def __init__(self, feature_size):
    super(AttentionModule, self).__init__()
    self.feature_size = feature_size
    self.fc1 = nn.Linear(self.feature_size, 256)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(256, 1)
    self.sigmoid = nn.Sigmoid()
  
  def forward(self, feature_segments):
    ## B x T x 1024
    x = self.fc1(feature_segments)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.sigmoid(x)
    return x  ## B x T
  
  def l1_sparsity_loss(self, x):
    return torch.sum(x, dim=1)
